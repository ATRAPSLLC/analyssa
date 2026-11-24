//! Binding time: which values are known before the program runs.
//!
//! A value is [`BindingTime::Static`] when it is fixed by the program and its
//! image alone - the same every time control reaches its definition, whatever
//! the inputs - and [`BindingTime::Dynamic`] otherwise. A specializer may
//! compute a static value ahead of time and must leave a dynamic one to the
//! program. Everything is dynamic until shown static, because the failure this
//! analysis exists to prevent is one direction only: calling a dynamic value
//! static folds a wrong constant into the output, which is worse than
//! recovering nothing.
//!
//! # The rules
//!
//! The analysis is a least fixpoint over the SSA values, and each value is one
//! of two things: dynamic outright, or the join of the values it is computed
//! from.
//!
//! - **Dynamic outright**: a value nothing in the function defines (a parameter,
//!   a register live on entry), the result of a call the host says nothing
//!   about, and the result of any operation whose effects are not
//!   [`SsaEffectKind::Pure`] - it reads state the analysis does not model, or
//!   the runtime supplies it.
//! - **A pure operation** is the join of its operands.
//! - **A load** is the value that reaches it through memory. A store whose cell
//!   is exactly the load's hands over its value; anything that may touch the
//!   cell without covering it exactly - a partial or unknown-address store, a
//!   call, a barrier - makes the load dynamic. What reaches from before the
//!   function is the region's contents at entry
//!   ([`BindingTime::of_contents_at_entry`]): read-only image data is static,
//!   and every other region may hold anything. A load of *contents* - rather
//!   than of a value a store handed over - is also as static as the index it
//!   selects its cell by: `table[i]` is a constant only when `i` is.
//! - **A merge**, whether a phi or a memory phi, is the join of what it merges
//!   and of the branches that decide which of them arrives. `x = c ? 1 : 2` is
//!   not a constant when `c` is not, though both arms are.
//!
//! The deciding branches are over-approximated: every conditional transfer on a
//! path from the merge's immediate dominator to the merge, including the
//! dominator's own and, for a loop header, every one inside the loop. A block
//! that may throw on such a path decides by whether it threw, which is dynamic.
//! A decoder loop whose bound and key are static stays static under this rule,
//! and anything the rule cannot vouch for is dynamic.
//!
//! # Regions
//!
//! Memory comes into it through Memory SSA ([`MemorySsa`](crate::analysis::memory::MemorySsa)), built with the
//! same [`AnalysisContext`]. Its budget is this analysis's: an exhausted build
//! classifies no load, so every load is dynamic, and [`BindingTimes::exhaustion`]
//! says so.
//!
//! # What a host adds
//!
//! Two things are facts about the program the analysis cannot derive from one
//! function, and a host states them through a [`BindingOracle`]:
//!
//! - **What a call does** ([`CallBinding`]): nothing known; a result that is a
//!   function of the arguments, such as a string decryptor or a helper whose
//!   body an interprocedural summary has read; or contents the call writes into
//!   an object it is handed, fixed by its other arguments, such as CIL's
//!   `RuntimeHelpers.InitializeArray` filling an array from a field's RVA data.
//! - **What a location holds on entry**: a static field the host knows is set
//!   once, from constants, before any code that reads it.
//!
//! The oracle `()` knows nothing, and gives exactly the rules above.
//!
//! [`SsaEffectKind::Pure`]: crate::ir::ops::effects::SsaEffectKind::Pure

use crate::{
    analysis::{
        budget::Exhaustion,
        context::AnalysisContext,
        dataflow::lattice::{JoinSemiLattice, Lattice, MeetSemiLattice},
        demand::Dependencies,
        memory::{MemoryLocation, MemoryRegion},
    },
    ir::{function::SsaFunction, variable::SsaVarId},
    target::Target,
};

/// When a value is known: before the program runs, or only as it runs.
///
/// Ordered `Static` below `Dynamic`: the join of two binding times is the later
/// one, and [`Dynamic`](Self::Dynamic) is the top - what the analysis answers
/// when it knows nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BindingTime {
    /// Fixed by the program and its image alone.
    Static,
    /// Depends on something only the running program has.
    Dynamic,
}

impl BindingTime {
    /// The binding time of a region's contents when a function is entered.
    ///
    /// Only read-only image data is static there: a write to it faults, so its
    /// bytes are the image's in every function. A writable region may have been
    /// written by anything that ran before, and a frame or an unknown region may
    /// hold anything a caller left in it.
    ///
    /// # Arguments
    ///
    /// * `region` - The region a location lies in.
    #[must_use]
    pub const fn of_contents_at_entry(region: MemoryRegion) -> Self {
        match region {
            MemoryRegion::ImageReadOnly => Self::Static,
            MemoryRegion::ImageWritable | MemoryRegion::Frame | MemoryRegion::Unknown => {
                Self::Dynamic
            }
        }
    }
}

impl MeetSemiLattice for BindingTime {
    fn meet(&self, other: &Self) -> Self {
        (*self).min(*other)
    }

    fn is_bottom(&self) -> bool {
        *self == Self::Static
    }
}

impl JoinSemiLattice for BindingTime {
    fn join(&self, other: &Self) -> Self {
        (*self).max(*other)
    }

    fn is_top(&self) -> bool {
        *self == Self::Dynamic
    }
}

impl Lattice for BindingTime {
    fn top() -> Self {
        Self::Dynamic
    }

    fn bottom() -> Self {
        Self::Static
    }
}

/// What a call does, as far as binding time is concerned.
///
/// Stated by a host through [`BindingOracle::call`]. Every answer is a promise
/// the analysis builds constants on, so a host that is unsure answers
/// [`Opaque`](Self::Opaque).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CallBinding {
    /// Nothing is known: the result is dynamic, and so is everything the call
    /// may write.
    #[default]
    Opaque,
    /// The result is fixed by ordinary arguments and implicit inputs: the same
    /// input values give the same result on every call, whatever else the
    /// program has done. This promise includes any pointee or other state the
    /// result depends on; a static pointer alone does not make its contents static.
    ///
    /// Says nothing about memory. The call still clobbers whatever it may
    /// write, so a decryptor that caches what it returns is still this.
    OfArguments,
    /// The call fills the object passed as argument `argument` with contents
    /// its other ordinary arguments and implicit inputs fix, and nothing else
    /// it does is known.
    ///
    /// Only the cells of that object - the locations whose base object is the
    /// value passed - take the call's arguments as their binding time; every
    /// other location is clobbered as by an opaque call. The object's own
    /// identity does not count: an array filled from a field's initial data
    /// holds the same elements wherever it was allocated.
    Fills {
        /// The ordinary argument index the object is passed as, excluding implicit inputs.
        argument: usize,
    },
}

/// What a host knows about binding time that one function does not say.
///
/// Both methods default to knowing nothing, and `()` implements the trait
/// that way.
pub trait BindingOracle<T: Target> {
    /// What a call of `method` does.
    ///
    /// Asked for direct and virtual calls. For a virtual call `method` is the
    /// method named at the call site, and the answer must hold for every
    /// override dispatch can reach.
    ///
    /// # Arguments
    ///
    /// * `method` - The method the call site names.
    fn call(&self, _method: &T::MethodRef) -> CallBinding {
        CallBinding::Opaque
    }

    /// The binding time of what `location` holds when a function is entered,
    /// or `None` to leave it to the location's region.
    ///
    /// A host answering [`BindingTime::Static`] promises the location holds the
    /// same value on every entry to every function that reads it - a field set
    /// once, from constants, before any reader runs.
    ///
    /// # Arguments
    ///
    /// * `location` - The location, as Memory SSA names it.
    fn contents_at_entry(&self, _location: &MemoryLocation<T>) -> Option<BindingTime> {
        None
    }
}

/// The oracle that knows nothing.
impl<T: Target> BindingOracle<T> for () {}

/// The binding time of every value in one function.
#[derive(Debug, Clone)]
pub struct BindingTimes {
    /// Each value's binding time, indexed by its id. An id past the end is a
    /// value this function does not define, which is dynamic.
    times: Vec<BindingTime>,
    /// The budget Memory SSA ran out of, when it did.
    exhaustion: Option<Exhaustion>,
}

impl BindingTimes {
    /// Computes the binding time of every value `ssa` defines.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function.
    /// * `context` - The pointer width, the image, and the budget Memory SSA is
    ///   built within.
    /// * `oracle` - What the host knows about calls and initial contents; `&()`
    ///   when it knows nothing.
    #[must_use]
    pub fn analyze<T: Target>(
        ssa: &SsaFunction<T>,
        context: &AnalysisContext<'_>,
        oracle: &dyn BindingOracle<T>,
    ) -> Self {
        let dependencies = Dependencies::analyze(ssa, context, oracle);
        let exhaustion = dependencies.exhaustion();
        Self {
            times: dependencies.binding_times(),
            exhaustion,
        }
    }

    /// Returns the binding time of `value`.
    ///
    /// A value this function does not define is dynamic.
    #[must_use]
    pub fn of(&self, value: SsaVarId) -> BindingTime {
        self.times
            .get(value.index())
            .copied()
            .unwrap_or(BindingTime::Dynamic)
    }

    /// Returns `true` when `value` is known before the program runs.
    #[must_use]
    pub fn is_static(&self, value: SsaVarId) -> bool {
        self.of(value) == BindingTime::Static
    }

    /// Returns every static value, in id order.
    pub fn static_values(&self) -> impl Iterator<Item = SsaVarId> + '_ {
        self.times
            .iter()
            .enumerate()
            .filter(|(_, time)| **time == BindingTime::Static)
            .map(|(index, _)| SsaVarId::from_index(index))
    }

    /// Returns the budget Memory SSA ran out of, when it did.
    ///
    /// Every load is then dynamic: nothing established which store it reads.
    #[must_use]
    pub const fn exhaustion(&self) -> Option<Exhaustion> {
        self.exhaustion
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{MACHINE, const_i32_return};

    /// The lattice is the two-point chain: `Static` below `Dynamic`.
    #[test]
    fn static_is_below_dynamic() {
        let (s, d) = (BindingTime::Static, BindingTime::Dynamic);
        assert_eq!(s.join(&d), d);
        assert_eq!(d.join(&s), d);
        assert_eq!(s.meet(&d), s);
        assert_eq!(BindingTime::top(), d);
        assert_eq!(BindingTime::bottom(), s);
        assert!(d.is_top() && s.is_bottom());
    }

    /// **The static values are those whose binding time is static.** A
    /// function returning a constant computes it before the program runs.
    #[test]
    fn the_static_values_are_the_ones_known_before_the_run() {
        let ssa = const_i32_return(3);
        let times = BindingTimes::analyze(&ssa, &AnalysisContext::new(MACHINE), &());
        let statics = times.static_values().collect::<Vec<_>>();
        assert!(!statics.is_empty());
        for index in 0..ssa.variable_count() {
            let value = SsaVarId::from_index(index);
            assert_eq!(statics.contains(&value), times.is_static(value));
        }
    }

    /// Only read-only image data enters a function static.
    #[test]
    fn only_read_only_image_data_is_static_at_entry() {
        assert_eq!(
            BindingTime::of_contents_at_entry(MemoryRegion::ImageReadOnly),
            BindingTime::Static
        );
        for region in [
            MemoryRegion::ImageWritable,
            MemoryRegion::Frame,
            MemoryRegion::Unknown,
        ] {
            assert_eq!(
                BindingTime::of_contents_at_entry(region),
                BindingTime::Dynamic
            );
        }
    }
}
