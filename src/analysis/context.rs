//! The facts about one host's target an analysis needs and cannot derive.
//!
//! [`SsaFunction`](crate::ir::SsaFunction) is generic over its [`Target`] but
//! carries no *instance* of it: `T` is how the crate names a host's types, not
//! a value holding the host's facts. So a fact that belongs to the running
//! host - the machine its code runs on ([`Machine`]), what its image holds,
//! how much work an analysis may do - cannot be asked of the function. It has
//! to be handed in.
//!
//! [`AnalysisContext`] is those facts in one value, borrowed for the length of
//! an analysis. A pass receives one through its
//! [`PassEnv`](crate::scheduling::PassEnv); a caller running an analysis
//! directly builds one with [`AnalysisContext::new`] and adds what it has.
//!
//! [`Target`]: crate::target::Target
//! [`Machine`]: crate::analysis::Machine

use std::{fmt, ops::Range};

use crate::{
    analysis::{budget::WorkBudget, consts::Machine, execution::LibraryModel, image::ImageData},
    interrupt::{Interrupt, NeverInterrupted},
};

/// What a host knows the functions its code calls do ([`LibraryModel`]), by
/// the call.
///
/// The host names the function a call reaches and holds what its catalog
/// states of it; an analysis of one function knows neither. A call this
/// answers `None` for does what nothing states.
pub trait CallModels: fmt::Debug + Send + Sync {
    /// The model of the function the call at `instruction` of `block` calls,
    /// where the host states one.
    fn model(&self, block: usize, instruction: usize) -> Option<LibraryModel>;
}

/// Target-instance facts, the image oracle, and the work budget.
#[derive(Clone, Copy)]
pub struct AnalysisContext<'a> {
    /// The machine the analysed code runs on, as the host states it.
    machine: Machine,
    /// What the image holds, when the host can say.
    image: Option<&'a dyn ImageData>,
    /// How much work an analysis may do.
    budget: WorkBudget,
    /// Attested disjoint numeric ranges of the current execution root.
    disjoint_numeric: &'a [Range<u64>],
    /// What the functions the analysed code calls do, when the host can say.
    models: Option<&'a dyn CallModels>,
    /// What the analysis asks whether it must stop, when the host can be
    /// stopped ([`Self::with_interrupt`]).
    interrupt: Option<&'a dyn Interrupt>,
}

impl<'a> AnalysisContext<'a> {
    /// A context of `machine`, with no image and the default budgets.
    #[must_use]
    pub const fn new(machine: Machine) -> Self {
        Self {
            machine,
            image: None,
            budget: WorkBudget::DEFAULT,
            disjoint_numeric: &[],
            models: None,
            interrupt: None,
        }
    }

    /// Returns this context with `models` as what the called functions do.
    #[must_use]
    pub const fn with_models(mut self, models: &'a dyn CallModels) -> Self {
        self.models = Some(models);
        self
    }

    /// What the functions the analysed code calls do, where the host said.
    #[must_use]
    pub const fn models(&self) -> Option<&'a dyn CallModels> {
        self.models
    }

    /// Returns this context with `interrupt` as what its analyses ask whether
    /// they must stop: a pipeline asks before each pass, and one stopped
    /// reports it rather than a result
    /// ([`PipelineReport::interrupted`](crate::scheduling::PipelineReport::interrupted)).
    #[must_use]
    pub const fn with_interrupt(mut self, interrupt: &'a dyn Interrupt) -> Self {
        self.interrupt = Some(interrupt);
        self
    }

    /// What the analysis asks whether it must stop:
    /// [`NeverInterrupted`] where the host gave no interrupt.
    #[must_use]
    pub fn interrupt(&self) -> &'a dyn Interrupt {
        self.interrupt.unwrap_or(&NeverInterrupted)
    }

    /// Whether the host wants the analysis to stop now; never where it gave
    /// no interrupt.
    #[must_use]
    pub fn interrupted(&self) -> bool {
        self.interrupt.is_some_and(Interrupt::interrupted)
    }

    /// Returns this context with `image` as its image oracle.
    #[must_use]
    pub const fn with_image(mut self, image: &'a dyn ImageData) -> Self {
        self.image = Some(image);
        self
    }

    /// Borrows the attested numeric ranges for one execution.
    pub fn with_disjoint_numeric<'range>(
        self,
        ranges: &'range [Range<u64>],
    ) -> AnalysisContext<'range>
    where
        'a: 'range,
    {
        AnalysisContext {
            machine: self.machine,
            image: self.image,
            budget: self.budget,
            disjoint_numeric: ranges,
            models: self.models,
            interrupt: self.interrupt,
        }
    }

    /// Borrows this execution's disjoint numeric ranges.
    pub const fn disjoint_numeric(&self) -> &[Range<u64>] {
        self.disjoint_numeric
    }

    /// Returns this context with `budget` as its work budget.
    #[must_use]
    pub const fn with_budget(mut self, budget: WorkBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Returns the machine the analysed code runs on: as the host states it,
    /// its byte order where it states none the order the image stores its
    /// values in.
    ///
    /// An analysis that reads memory a byte at a time needs the byte order to
    /// take a byte of a wider value, and answers nothing where neither the
    /// host nor an image says rather than guess.
    #[must_use]
    pub fn machine(&self) -> Machine {
        Machine {
            byte_order: self
                .machine
                .byte_order
                .or_else(|| self.image.map(|image| image.endianness())),
            ..self.machine
        }
    }

    /// Returns the image oracle, when the host supplied one.
    #[must_use]
    pub const fn image(&self) -> Option<&'a dyn ImageData> {
        self.image
    }

    /// Returns the work budget.
    #[must_use]
    pub const fn budget(&self) -> &WorkBudget {
        &self.budget
    }
}

impl fmt::Debug for AnalysisContext<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnalysisContext")
            .field("machine", &self.machine)
            .field("has_image", &self.image.is_some())
            .field("has_models", &self.models.is_some())
            .field("budget", &self.budget)
            .finish()
    }
}
