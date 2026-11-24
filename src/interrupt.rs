//! Stopping an analysis from outside.
//!
//! An analysis is a long stretch of pure CPU work: over a large image a
//! whole-program flow, or the normalization of one large function, runs for
//! seconds inside a single call, with no suspension point a caller's timeout
//! could ever land on. The probe here is what the work itself consults, at
//! phase, pass and item boundaries, so a host can bound it
//! ([`AnalysisContext::with_interrupt`](crate::analysis::AnalysisContext::with_interrupt)).
//!
//! **A tripped interrupt aborts; it never truncates.** Discovery is bounded by
//! iteration counts and never by a clock: a shorter listing produced because the
//! box was busy would make a content-addressed identity depend on load, while
//! *no* listing does not. So the only thing an interrupt can cause is the whole
//! run failing with [`Interrupted`], which the caller reports as a failed unit
//! of work, and whatever a phase left half done is discarded with it.

use std::{
    error, fmt,
    sync::atomic::{AtomicBool, Ordering},
};

/// Asked at phase and item boundaries whether the run must stop.
///
/// Implementors are consulted often - per phase, per round of a fixpoint and
/// per function of a whole-image loop - so the check must be cheap: an atomic
/// load, not a syscall. Once tripped, an interrupt stays tripped: a phase that
/// sees it late still stops.
pub trait Interrupt: Sync {
    /// Whether the run must stop now.
    fn interrupted(&self) -> bool;

    /// Returns [`Interrupted`] when the run must stop now
    /// ([`Self::interrupted`]), for a phase to return with `?`.
    ///
    /// # Errors
    ///
    /// [`Interrupted`] when the probe has tripped.
    fn check(&self) -> Result<(), Interrupted> {
        if self.interrupted() {
            Err(Interrupted)
        } else {
            Ok(())
        }
    }
}

/// The interrupt for a caller that has nothing to cancel against.
///
/// The default, so a fixture harness or a test driver builds an engine without
/// inventing a probe.
#[derive(Debug, Clone, Copy, Default)]
pub struct NeverInterrupted;

impl Interrupt for NeverInterrupted {
    fn interrupted(&self) -> bool {
        false
    }
}

/// A run that stopped because its [`Interrupt`] tripped.
///
/// Carries no partial output by construction: the whole point is that a
/// cancelled run produces nothing rather than something shorter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interrupted;

impl fmt::Display for Interrupted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the analysis was interrupted before it completed")
    }
}

impl error::Error for Interrupted {}

/// An [`Interrupt`] a parallel loop shares, latched once it trips.
///
/// Rayon has no abort: once a loop has started, every item is still visited.
/// Each item asks [`Self::stopped`] first and, once the probe has tripped,
/// returns at once without its work; the loop's result is then thrown away by
/// [`Self::finish`]. Filtering the stopped items out instead would build a
/// *shorter* result, the one outcome an interrupt must never produce.
pub struct Latch<'a> {
    /// The probe.
    interrupt: &'a dyn Interrupt,
    /// Whether an item has seen the probe tripped.
    tripped: AtomicBool,
}

impl<'a> Latch<'a> {
    /// A latch over `interrupt`, not yet tripped.
    #[must_use]
    pub fn new(interrupt: &'a dyn Interrupt) -> Self {
        Self {
            interrupt,
            tripped: AtomicBool::new(false),
        }
    }

    /// Whether the loop must stop: the latch has tripped, or the probe has
    /// just tripped and latches it.
    #[must_use]
    pub fn stopped(&self) -> bool {
        if self.tripped.load(Ordering::Relaxed) {
            return true;
        }
        if self.interrupt.interrupted() {
            self.tripped.store(true, Ordering::Relaxed);
            return true;
        }
        false
    }

    /// Ends the loop: [`Interrupted`] where any item stopped, or where the
    /// probe tripped inside an item after it began, so the partial result is
    /// discarded.
    ///
    /// # Errors
    ///
    /// [`Interrupted`] when the latch or the probe tripped during the loop.
    pub fn finish(self) -> Result<(), Interrupted> {
        if self.tripped.load(Ordering::Relaxed) || self.interrupt.interrupted() {
            Err(Interrupted)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use super::*;

    /// An interrupt that trips on its `after`th poll and stays tripped.
    struct After {
        /// The poll it trips on, counting from one.
        after: usize,
        /// Polls so far.
        polls: AtomicUsize,
    }

    impl Interrupt for After {
        fn interrupted(&self) -> bool {
            self.polls.fetch_add(1, Ordering::Relaxed).saturating_add(1) >= self.after
        }
    }

    /// **A latch that trips stays tripped**: the item that sees the probe trip
    /// and every item after it stop, without asking the probe again, and the
    /// loop ends interrupted; a latch that never tripped ends clean.
    #[test]
    fn a_latch_that_trips_stays_tripped() {
        let probe = After {
            after: 3,
            polls: AtomicUsize::new(0),
        };
        let latch = Latch::new(&probe);
        let stopped: Vec<bool> = (0..6).map(|_| latch.stopped()).collect();
        assert_eq!(stopped, [false, false, true, true, true, true]);
        assert_eq!(
            probe.polls.load(Ordering::Relaxed),
            3,
            "asked until it tripped"
        );
        assert_eq!(latch.finish(), Err(Interrupted));
        let clean = Latch::new(&NeverInterrupted);
        assert!(!clean.stopped());
        assert_eq!(clean.finish(), Ok(()));
        assert_eq!(
            probe.check(),
            Err(Interrupted),
            "a tripped probe checks as stopped"
        );
        assert_eq!(NeverInterrupted.check(), Ok(()));
    }
}
