//! The values a call passes, captured before control leaves the caller.
//!
//! A call hands its callee ordinary arguments, implicit machine inputs keyed by
//! their role, and stack values staged for it ([`CallInputs`]). They are
//! captured once, before the call transfers control or retires its frame, so a
//! callee binding, a model and an unknown-effect clobber all see the same
//! values. Implicit inputs are found by key through [`InputIndex`], inline for
//! the usual one or two and in bounded tree storage beyond that.

use std::{collections::BTreeMap, mem};

use smallvec::SmallVec;

use crate::{
    analysis::execution::{
        ModelCall,
        access::Access,
        arguments::{ArgumentProjection, PhysicalArgument},
        state::State,
        trace::Trace,
        value::{Address, Value},
    },
    ir::{CallInput, CallInputKey, SsaVarId, StagedValue},
    target::Target,
};

/// A small inline lookup, with bounded tree storage for larger host contracts.
///
/// Indices address only the confirmed implicit range. Repeated keys resolve to their
/// last occurrence without dropping any declared SSA use or captured value.
pub enum InputIndex {
    /// At most two entries, searched in reverse emission order.
    Inline(SmallVec<[(CallInputKey, usize); 2]>),
    /// Unique keys mapped to their last suffix position.
    Indexed(BTreeMap<CallInputKey, usize>),
}

impl InputIndex {
    /// Builds an inline or indexed lookup, retaining the last position of repeated keys.
    fn new(inputs: &[CallInput]) -> Self {
        if inputs.len() <= 2 {
            Self::Inline(
                inputs
                    .iter()
                    .enumerate()
                    .map(|(index, input)| (input.key, index))
                    .collect(),
            )
        } else {
            Self::Indexed(
                inputs
                    .iter()
                    .enumerate()
                    .map(|(index, input)| (input.key, index))
                    .collect(),
            )
        }
    }

    /// Returns the last implicit position for a semantic key.
    pub fn get(&self, key: CallInputKey) -> Option<usize> {
        match self {
            Self::Inline(entries) => entries
                .iter()
                .rev()
                .find_map(|(held, index)| (*held == key).then_some(*index)),
            Self::Indexed(entries) => entries.get(&key).copied(),
        }
    }
}

/// Borrows confirmed machine inputs captured before call selection or frame retirement.
///
/// The host can inspect an explicitly declared semantic input without reading
/// caller registers again, projecting ABI arguments, or changing execution state.
/// Values remain owned by this call's single input capture.
pub struct CallValues<'a, T: Target> {
    /// Original captured call values and their existing semantic-key index.
    inputs: &'a CallInputs<T>,
}

impl<T: Target> CallValues<'_, T> {
    /// Returns the last confirmed implicit value carrying the host's semantic key.
    #[must_use]
    pub fn implicit(&self, key: CallInputKey) -> Option<&Value<T>> {
        let (values, index) = self.inputs.implicit();
        values.get(index.get(key)?)
    }
}

/// Actual values captured before the call transfers control or retires its frame.
///
/// A single vector holds ordinary arguments, implicit values and staged stack
/// candidates in separate ranges. Only ordinary arguments and implicit values
/// bind to callee inputs. Staged values retain possible aliases for unknown
/// effects and returns without asserting that the callee reads them.
pub struct CallInputs<T: Target> {
    /// Captured ordinary, implicit and staged values in emission order.
    values: Vec<Value<T>>,
    /// Length of the ordinary prefix.
    ordinary_count: usize,
    /// Exclusive end of confirmed implicit inputs, before staged candidates.
    implicit_end: usize,
    /// Whether any captured source may carry private stack-pointer bits.
    stack_aliases: bool,
    /// Last relative position of each semantic key within the implicit range.
    index: InputIndex,
    /// Actual carriers for an explicit host-bound ordinary argument view.
    physical: Option<Vec<PhysicalArgument<T>>>,
    /// The start of the object each ordinary argument is a pointer into,
    /// where it is one.
    objects: Vec<Option<Address>>,
}

impl<T: Target> CallInputs<T> {
    /// Captures each call input once before control transfers or the frame retires.
    ///
    /// Returns `None` when the combined input domains cannot be indexed by `usize`.
    pub fn capture(
        ordinary: &[SsaVarId],
        implicit: &[CallInput],
        staged: &[StagedValue],
        state: &State<T>,
    ) -> Option<Self> {
        let (implicit_end, count) =
            ordinary
                .len()
                .checked_add(implicit.len())
                .and_then(|implicit_end| {
                    implicit_end
                        .checked_add(staged.len())
                        .map(|count| (implicit_end, count))
                })?;
        let index = InputIndex::new(implicit);
        let mut values = Vec::with_capacity(count);
        let mut stack_aliases = false;
        for value in ordinary
            .iter()
            .copied()
            .chain(implicit.iter().map(|input| input.value))
            .chain(staged.iter().map(|input| input.value))
        {
            let held = state.value(value);
            stack_aliases |= held.may_reference_stack();
            values.push(held);
        }
        Some(Self {
            values,
            ordinary_count: ordinary.len(),
            implicit_end,
            stack_aliases,
            index,
            physical: None,
            objects: Vec::new(),
        })
    }

    /// States the start of the object each ordinary argument is a pointer
    /// into, in argument order, `None` where it is none.
    #[must_use]
    pub fn with_objects(mut self, objects: Vec<Option<Address>>) -> Self {
        self.objects = objects;
        self
    }

    /// The start of the object an ordinary argument holding `value` is a
    /// pointer into, where one is.
    pub fn object_of(&self, value: &Value<T>) -> Option<Address> {
        self.ordinary()
            .iter()
            .zip(&self.objects)
            .find_map(|(argument, object)| (argument == value).then_some(*object).flatten())
    }

    /// Borrows captured source inputs before selecting a provider or projecting arguments.
    pub fn view(&self) -> CallValues<'_, T> {
        CallValues { inputs: self }
    }

    /// Returns the ordinary argument prefix, excluding every implicit value.
    pub fn ordinary(&self) -> &[Value<T>] {
        self.values.get(..self.ordinary_count).unwrap_or(&[])
    }

    /// Returns a uniquely identified staged value from the captured suffix.
    pub fn staged_at(&self, staged: &[StagedValue], offset: i64) -> Option<&Value<T>> {
        let mut found = staged
            .iter()
            .enumerate()
            .filter(|(_, value)| value.offset == offset);
        let (index, _) = found.next()?;
        if found.next().is_some() {
            return None;
        }
        self.values.get(self.implicit_end.checked_add(index)?)
    }

    /// Installs one host-bound logical view while retaining every original alias source.
    pub fn project(&mut self, projection: ArgumentProjection<T>) {
        let Some(mut values) = projection.values else {
            return;
        };
        let mut original = mem::take(&mut self.values);
        let mut suffix = original.split_off(self.ordinary_count);
        let ordinary_count = values.len();
        let implicit_count = self.implicit_end.saturating_sub(self.ordinary_count);
        values.append(&mut suffix);
        values.append(&mut original);
        self.stack_aliases |= values.iter().any(Value::may_reference_stack)
            || projection
                .physical
                .iter()
                .any(|value| value.value.may_reference_stack());
        self.values = values;
        self.ordinary_count = ordinary_count;
        self.implicit_end = ordinary_count.saturating_add(implicit_count);
        self.physical = Some(projection.physical);
    }

    /// Returns whether logical positions replace the source operand indices.
    pub const fn is_projected(&self) -> bool {
        self.physical.is_some()
    }

    /// Borrows actual carrier values in the logical argument order.
    pub fn physical(&self) -> Option<&[PhysicalArgument<T>]> {
        self.physical.as_deref()
    }

    /// Returns cached stack provenance gathered while capturing inputs.
    pub const fn may_reference_stack(&self) -> bool {
        self.stack_aliases
    }

    /// Returns all possible inputs for escape, alias and unknown-effect handling.
    pub fn all(&self) -> &[Value<T>] {
        &self.values
    }

    /// Returns only confirmed implicit inputs and their lookup for callee binding.
    pub fn implicit(&self) -> (&[Value<T>], &InputIndex) {
        (
            self.values
                .get(self.ordinary_count..self.implicit_end)
                .unwrap_or(&[]),
            &self.index,
        )
    }

    /// Opens one alternative of a model's call on these inputs, on the path
    /// `state` and `trace` record: its own copy of the captured values, the
    /// implicit inputs' lookup, and the object each ordinary argument is a
    /// pointer into.
    pub fn model_call<'a>(
        &'a self,
        state: &'a mut State<T>,
        trace: &'a mut Trace,
        access: Access<'a>,
    ) -> ModelCall<'a, T> {
        ModelCall::new(self.values.clone(), state, trace, access)
            .with_implicit_inputs(self.ordinary_count, &self.index)
            .with_objects(&self.objects)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        analysis::execution::frame::Frame,
        ir::{
            ConstValue, SsaFunction,
            variable::{DefSite, VariableOrigin},
        },
        pointer::PointerSize,
        testing::{MockTarget, MockType},
    };

    /// Supplies distinct SSA uses while allowing repeated semantic keys.
    fn inputs(keys: &[u32]) -> Vec<CallInput> {
        keys.iter()
            .enumerate()
            .map(|(index, key)| CallInput {
                key: CallInputKey::new(*key),
                value: SsaVarId::from_index(index),
            })
            .collect()
    }

    /// Inline and tree lookups agree with an independent reverse-search oracle.
    #[test]
    fn implicit_index_last_key_wins_in_both_storage_forms() {
        for keys in [vec![], vec![0], vec![0, 0], vec![17, 0, u32::MAX, 17]] {
            let index = InputIndex::new(&inputs(&keys));
            for key in [0, 17, 9, u32::MAX] {
                assert_eq!(
                    index.get(CallInputKey::new(key)),
                    keys.iter().rposition(|candidate| *candidate == key)
                );
            }
        }
    }

    /// Capturing a hidden owned value preserves its payload and the borrowed caller.
    #[test]
    fn implicit_capture_preserves_owned_payloads_and_caller() {
        let mut body = SsaFunction::<MockTarget>::new(0, 1);
        let value = body.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::Ref,
        );
        let state = State::new(Frame::enter(
            &body,
            0,
            0,
            0,
            vec![(
                value,
                Value::Const(ConstValue::DecryptedString("12345678".into())),
            )],
            PointerSize::Bit64,
        ));
        let implicit = [CallInput {
            key: CallInputKey::new(17),
            value,
        }];
        let captured = CallInputs::capture(&[], &implicit, &[], &state).unwrap();
        assert!(captured.ordinary().is_empty());
        assert_eq!(captured.all().len(), 1);
        assert_eq!(
            captured.implicit().0[0].constant(),
            Some(&ConstValue::DecryptedString("12345678".into()))
        );
        assert_eq!(
            state.value(value).constant(),
            captured.implicit().0[0].constant()
        );
    }

    /// Staged pointers retain aliases without occupying an argument or semantic key.
    #[test]
    fn staged_capture_keeps_aliases_outside_confirmed_input_domains() {
        let mut body = SsaFunction::<MockTarget>::new(0, 1);
        let value = body.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        let state = State::new(Frame::enter(
            &body,
            0,
            0,
            0,
            vec![(value, Value::StackDerived)],
            PointerSize::Bit64,
        ));
        let staged = [StagedValue { offset: -8, value }];
        let captured = CallInputs::capture(&[], &[], &staged, &state).unwrap();
        assert!(captured.ordinary().is_empty());
        assert!(captured.implicit().0.is_empty());
        assert_eq!(captured.implicit().1.get(CallInputKey::new(0)), None);
        assert_eq!(captured.all().len(), 1);
        assert!(captured.may_reference_stack());
    }
}
