//! Finite pointer-use extents collected during the execution that establishes facts.
//!
//! Access extents union across visits; byte observations intersect separately.
//! This bounds a call's uses and never establishes an allocation or its lifetime.

use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

use crate::analysis::execution::{
    facts::{ArgumentBytes, UseBound},
    value::{Address, Root},
};

/// An observed call in its caller's exact retained source domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct UseSite {
    /// The retained caller, or the original root.
    pub owner: Option<usize>,
    /// Instruction coordinates in that caller.
    pub position: (usize, usize),
}

/// Accesses possibly belonging to one observed pointer argument.
#[derive(Debug)]
struct Extent {
    /// Exact pre-call pointer coordinate.
    pointer: Address,
    /// A finite upper bound, absent after any unresolved relevant access.
    bytes: Option<usize>,
}

impl Extent {
    /// Includes a complete linear memory access without inferring object identity.
    fn include(&mut self, address: Address, length: usize) {
        let Some(previous) = self.bytes else {
            return;
        };
        if length == 0 {
            return;
        }
        let own = self.pointer.root();
        let other = address.root();
        if own != other {
            // Concrete image, activation and allocation roots are disjoint.
            // Integer and opaque coordinates may still name those objects.
            let concrete = |root| {
                matches!(
                    root,
                    Root::Symbol | Root::Stack | Root::StackFrame(_) | Root::Heap(_)
                )
            };
            if !concrete(own) || !concrete(other) {
                self.bytes = None;
            }
            return;
        }
        if address.pointer_size() != self.pointer.pointer_size()
            || address.linear_prefix_len(length) != length
        {
            self.bytes = None;
            return;
        }
        let coordinate = |at: Address| {
            if at.root().unsigned_offsets() {
                i128::from(at.offset().cast_unsigned())
            } else {
                i128::from(at.offset())
            }
        };
        let end = coordinate(address).saturating_add(length as i128);
        let extent = end.saturating_sub(coordinate(self.pointer));
        if extent > 0 {
            self.bytes = usize::try_from(extent)
                .ok()
                .map(|extent| previous.max(extent));
        }
    }
}

/// Complete coverage and candidate pointer extents for one observed call.
#[derive(Debug, Default)]
struct CallUses {
    /// An unentered or unsupported visit cannot establish a complete use bound.
    unresolved: bool,
    /// Candidates keyed by their original argument positions.
    arguments: BTreeMap<usize, Extent>,
}

/// Invocation-local access unions, shared across alternative paths.
#[derive(Debug, Default)]
pub struct Uses {
    /// Exact caller-domain/site keys; no payload hash substitutes for equality.
    calls: BTreeMap<UseSite, CallUses>,
}

impl Uses {
    /// Admits another visit without resetting any earlier access or refusal.
    fn enter(&mut self, site: UseSite, arguments: &[(usize, Address)]) {
        let fresh = !self.calls.contains_key(&site);
        let call = self.calls.entry(site).or_default();
        for (argument, extent) in &mut call.arguments {
            if arguments
                .binary_search_by_key(argument, |(index, _)| *index)
                .ok()
                .and_then(|slot| arguments.get(slot))
                .is_none_or(|(_, pointer)| *pointer != extent.pointer)
            {
                extent.bytes = None;
            }
        }
        for &(argument, pointer) in arguments {
            call.arguments.entry(argument).or_insert(Extent {
                pointer,
                bytes: (fresh && !call.unresolved).then_some(0),
            });
        }
    }

    /// Withholds complete bounds for any unaccounted visit or effect.
    pub fn unresolved(&mut self, site: UseSite) {
        self.calls.entry(site).or_default().unresolved = true;
    }

    /// Restricts existing pre-call values to a completely covered use extent,
    /// which then bounds them as the callee's own ([`UseBound::Callee`]).
    pub fn restrict(&self, site: UseSite, arguments: &mut Vec<ArgumentBytes>) {
        let Some(call) = self.calls.get(&site).filter(|call| !call.unresolved) else {
            return;
        };
        for argument in arguments.iter_mut() {
            if let Some(extent) = call.arguments.get(&argument.argument)
                && extent.pointer == argument.address
                && let Some(bytes) = extent.bytes
            {
                argument.restrict(u64::try_from(bytes).unwrap_or(u64::MAX), UseBound::Callee);
            }
        }
        arguments.retain(|argument| !argument.bytes.is_empty());
    }
}

/// Active call obligations and their shared monotone coverage.
#[derive(Debug, Clone)]
pub struct UseTrace {
    /// Unioned evidence is not lost when an equal machine state is discarded.
    uses: Rc<RefCell<Uses>>,
    /// Tail transfers keep obligations at the same physical frame depth.
    active: Vec<(usize, UseSite)>,
    /// Whether this instruction supplied its complete supported destination span.
    wrote: bool,
}

impl PartialEq for UseTrace {
    /// Compares obligation handles without borrowing their shared mutable coverage.
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.uses, &other.uses)
            && self.active == other.active
            && self.wrote == other.wrote
    }
}

impl Eq for UseTrace {}

impl UseTrace {
    /// Starts a path with no active pointee-use obligation.
    pub fn new(uses: Rc<RefCell<Uses>>) -> Self {
        Self {
            uses,
            active: Vec::new(),
            wrote: false,
        }
    }

    /// Starts the actual call while preserving retired tail obligations.
    pub fn enter(&mut self, site: UseSite, depth: usize, arguments: &[(usize, Address)]) {
        self.uses.borrow_mut().enter(site, arguments);
        self.active.push((depth, site));
    }

    /// Ends every obligation implemented by this activation's actual return.
    pub fn returned(&mut self, depth: usize) {
        self.active.retain(|(entered, _)| *entered < depth);
    }

    /// Begins coverage of one instruction without altering accumulated uses.
    pub fn begin(&mut self) {
        self.wrote = false;
    }

    /// Returns whether the current instruction supplied a destination span.
    pub const fn wrote(&self) -> bool {
        self.wrote
    }

    /// Includes an exact access in every live obligation.
    pub fn access(&self, address: Address, length: usize) {
        let mut uses = self.uses.borrow_mut();
        for (_, site) in &self.active {
            if let Some(call) = uses.calls.get_mut(site).filter(|call| !call.unresolved) {
                for extent in call.arguments.values_mut() {
                    extent.include(address, length);
                }
            }
        }
    }

    /// Includes a complete supported destination span.
    pub fn write(&mut self, address: Address, length: usize) {
        self.wrote = true;
        self.access(address, length);
    }

    /// Withholds bounds after an access whose full footprint is unestablished.
    pub fn unresolved(&self) {
        let mut uses = self.uses.borrow_mut();
        for (_, site) in &self.active {
            uses.unresolved(*site);
        }
    }

    /// Preserves matching ancestry; different ancestry cannot certify complete coverage.
    pub fn join(&self, other: &Self) -> Option<Self> {
        if Rc::ptr_eq(&self.uses, &other.uses) && self.active == other.active {
            return Some(self.clone());
        }
        self.unresolved();
        other.unresolved();
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::execution::facts::{ByteBound, WrittenRanges},
        pointer::PointerSize,
    };

    fn pointer(root: Root, offset: i64) -> Address {
        Address::new(root, offset, PointerSize::Bit64)
    }

    #[test]
    fn alternatives_union_uses_instead_of_intersecting_their_lengths() {
        let uses = Rc::new(RefCell::new(Uses::default()));
        let site = UseSite {
            owner: None,
            position: (0, 1),
        };
        let address = pointer(Root::Stack, -16);
        let mut first = UseTrace::new(uses.clone());
        first.enter(site, 2, &[(0, address)]);
        let second = first.clone();
        first.access(address, 4);
        second.access(address, 8);
        let coverage = uses.borrow_mut();
        assert_eq!(
            first, second,
            "shared coverage is independent of local trace equality"
        );
        drop(coverage);
        let mut bytes = vec![ArgumentBytes {
            argument: 0,
            address,
            bytes: vec![7; 16],
            bound: ByteBound::Prefix,
            allocated_at: None,
            written: WrittenRanges::whole(16),
        }];
        uses.borrow().restrict(site, &mut bytes);
        assert_eq!(bytes[0].bytes.len(), 8);
        assert_eq!(
            bytes[0].bound,
            ByteBound::Use {
                bytes: 8,
                by: UseBound::Callee,
            },
            "the callee's own accesses bound the run"
        );
        second.unresolved();
        bytes[0].bytes = vec![7; 16];
        uses.borrow().restrict(site, &mut bytes);
        assert_eq!(
            bytes[0].bytes.len(),
            16,
            "unresolved coverage supplies no cap"
        );
    }

    #[test]
    fn concrete_disjoint_roots_and_prior_stack_slots_do_not_enlarge_a_use() {
        let mut extent = Extent {
            pointer: pointer(Root::Stack, -16),
            bytes: Some(0),
        };
        extent.include(pointer(Root::Stack, -32), 8);
        extent.include(pointer(Root::Heap(1), 0), 1024);
        extent.include(pointer(Root::Stack, -16), 4);
        assert_eq!(extent.bytes, Some(4));
        extent.include(pointer(Root::Number, 0x1000), 1);
        assert_eq!(extent.bytes, None, "unmapped integer coordinates can alias");
    }

    #[test]
    fn target_wrap_cannot_supply_a_linear_use_bound() {
        let address = Address::new(Root::Symbol, 0xfffe, PointerSize::Bit16);
        let mut extent = Extent {
            pointer: address,
            bytes: Some(0),
        };
        extent.include(address, 4);
        assert_eq!(extent.bytes, None);
    }

    #[test]
    fn tail_obligations_end_at_the_actual_return_depth() {
        let uses = Rc::new(RefCell::new(Uses::default()));
        let address = pointer(Root::Stack, -16);
        let outer = UseSite {
            owner: None,
            position: (0, 1),
        };
        let tail = UseSite {
            owner: Some(0),
            position: (0, 2),
        };
        let nested = UseSite {
            owner: Some(1),
            position: (0, 3),
        };
        let mut trace = UseTrace::new(uses.clone());
        trace.enter(outer, 2, &[(0, address)]);
        trace.enter(tail, 2, &[(0, address)]);
        trace.enter(nested, 3, &[(0, address)]);
        trace.returned(3);
        assert_eq!(trace.active, [(2, outer), (2, tail)]);
        trace.returned(2);
        assert!(trace.active.is_empty());
    }
}
