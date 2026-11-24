//! The C++ exceptions a run raises and handles, as an Itanium personality
//! routine keeps them: the one in flight while a raise unwinds, the ones
//! handlers have caught, and the selector the runtime handed the handler it
//! entered last.
//!
//! `__cxa_throw` puts an exception in flight and raises it; a landing pad the
//! raise enters receives the selector of the clause the personality chose
//! ([`SsaExceptionHandler::selector`]); `__cxa_begin_catch` moves the
//! exception from flight to the caught stack, `__cxa_end_catch` takes it off,
//! and `__cxa_rethrow` and `_Unwind_Resume` raise again what is caught or in
//! flight.
//!
//! [`SsaExceptionHandler::selector`]: crate::ir::SsaExceptionHandler::selector

use crate::{
    analysis::execution::value::{Address, Root, Value},
    target::Target,
};

/// One thrown exception: the object `__cxa_throw` was handed, the
/// `type_info` it names, and the destructor the runtime runs on the object
/// once the last handler is done with it.
#[derive(Debug, Clone, PartialEq)]
pub struct Thrown<T: Target> {
    /// The exception object.
    pub object: Value<T>,
    /// The module address of the `type_info` naming the object's type, where
    /// the run knows it.
    pub type_info: Option<u64>,
    /// The destructor, a function pointer or null.
    pub destructor: Value<T>,
}

impl<T: Target> Thrown<T> {
    /// The exception `__cxa_throw(object, type_info, destructor)` raises,
    /// `type_info` the address its argument names, where it names one.
    #[must_use]
    pub fn new(object: Value<T>, type_info: Option<Address>, destructor: Value<T>) -> Self {
        let type_info = type_info
            .filter(|address| address.root() == Root::Symbol)
            .and_then(|address| u64::try_from(address.offset()).ok());
        Self {
            object,
            type_info,
            destructor,
        }
    }

    /// Whether a catch clause naming the type at module address `caught`
    /// (`None` for `catch (...)`) takes this exception: `Some(true)` for a
    /// catch-all or the very type thrown; `None` otherwise, since a clause
    /// naming another type takes the exception where that type is a base of
    /// the thrown one, which the `type_info` objects state and the run does
    /// not read.
    #[must_use]
    pub fn caught_by(&self, caught: Option<u64>) -> Option<bool> {
        match caught {
            None => Some(true),
            Some(caught) => (self.type_info == Some(caught)).then_some(true),
        }
    }

    /// Whether the runtime runs no destructor on the object: its destructor
    /// is a known null.
    #[must_use]
    pub fn destroys_nothing(&self) -> bool {
        self.destructor
            .constant()
            .is_some_and(|value| value.is_zero())
    }
}

/// The run's C++ exceptions: the one in flight while a raise unwinds, the
/// ones handlers have caught, and the selector handed the handler a raise
/// entered last.
#[derive(Debug, Clone, PartialEq)]
pub struct Exceptions<T: Target> {
    /// The exception a raise is unwinding with, from its throw to the handler
    /// that catches it; through a cleanup, which resumes the unwind.
    in_flight: Option<Thrown<T>>,
    /// The exceptions handlers have caught and not yet finished with, the
    /// innermost last.
    caught: Vec<Thrown<T>>,
    /// The selector the runtime handed the handler a raise entered last.
    selector: Option<i64>,
}

impl<T: Target> Default for Exceptions<T> {
    fn default() -> Self {
        Self {
            in_flight: None,
            caught: Vec::new(),
            selector: None,
        }
    }
}

impl<T: Target> Exceptions<T> {
    /// The exception a raise is unwinding with.
    #[must_use]
    pub const fn in_flight(&self) -> Option<&Thrown<T>> {
        self.in_flight.as_ref()
    }

    /// Puts `thrown` in flight, as `__cxa_throw` does before it raises.
    pub fn throw(&mut self, thrown: Thrown<T>) {
        self.in_flight = Some(thrown);
    }

    /// Puts the innermost caught exception in flight again, as
    /// `__cxa_rethrow` does; `false` when no handler holds one.
    pub fn rethrow(&mut self) -> bool {
        match self.caught.last() {
            Some(caught) => {
                self.in_flight = Some(caught.clone());
                true
            }
            None => false,
        }
    }

    /// Takes the exception in flight into a handler, as `__cxa_begin_catch`
    /// does, and returns it.
    pub fn begin_catch(&mut self) -> Option<&Thrown<T>> {
        let thrown = self.in_flight.take()?;
        self.caught.push(thrown);
        self.caught.last()
    }

    /// Takes the innermost caught exception off, as `__cxa_end_catch` does,
    /// and returns it.
    pub fn end_catch(&mut self) -> Option<Thrown<T>> {
        self.caught.pop()
    }

    /// Hands `selector` to the handler the raise enters.
    pub fn deliver(&mut self, selector: Option<i64>) {
        self.selector = selector;
    }

    /// The selector the runtime handed the handler entered last.
    #[must_use]
    pub const fn selector(&self) -> Option<i64> {
        self.selector
    }
}
