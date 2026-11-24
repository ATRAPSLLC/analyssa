//! What a pass needs to run on one function.
//!
//! A pass that rewrites a function needs four things: somewhere to record what
//! it did, the host's facts about its target - pointer width, image, work
//! budget, together an [`AnalysisContext`] - which method it is looking at, and
//! sometimes the rest of the program. [`SsaPassHost`] carries all of that and
//! more, but it is a *whole-program* surface: implementing it means owning a
//! method store, a call graph and a dirty set. A pass that only rewrites the
//! function in front of it should not have to name such a type, and a host that
//! wants to run one pass over one function should not have to build one.
//!
//! [`PassEnv`] is the per-function surface, and [`FunctionPass`] is a pass
//! written against it. A `FunctionPass` is automatically an [`SsaPass`] too, so
//! the scheduler runs both kinds and a host with its own whole-program passes
//! keeps them.

use std::fmt;

use crate::{
    Error, Result,
    analysis::{
        bindingtime::BindingOracle,
        budget::{Exhaustion, WorkBudget},
        cache::FunctionAnalyses,
        context::AnalysisContext,
        image::ImageData,
    },
    events::{Event, EventBuilder, EventKind, EventListener},
    interrupt::Interrupt,
    ir::function::SsaFunction,
    passes::ranges::RangeFacts,
    scheduling::{
        capability::DeobfuscationCapability,
        pass::{ModificationScope, SsaPass, SsaPassHost},
    },
    target::Target,
    world::World,
};

/// Everything a pass needs about the function it is rewriting.
///
/// Borrowed rather than owned, and built either from a host
/// ([`SsaPassHost::env`]) or directly ([`PassEnv::new`]) by a caller running a
/// pass outside a scheduler.
pub struct PassEnv<'h, T: Target> {
    /// Where the pass records what it did.
    events: &'h dyn EventListener<T>,
    /// The host's target facts: its machine, image, work budget.
    context: AnalysisContext<'h>,
    /// What the host knows about calls and initial contents.
    binding: &'h dyn BindingOracle<T>,
    /// The method being rewritten.
    method: &'h T::MethodRef,
    /// The rest of the program, for a pass that reads its callees.
    world: Option<&'h dyn SsaPassHost<T>>,
    /// The method's range facts, shared by every pass run under this
    /// environment.
    ranges: RangeFacts,
}

impl<'h, T: Target> PassEnv<'h, T> {
    /// Builds an environment for one method, with no view of the program.
    ///
    /// For a caller running a pass over a function it holds: a test, a host
    /// applying one rewrite outside its pipeline, a tool. A pass that reads
    /// peer methods gets `None` from [`world`](Self::world) and must have a
    /// per-function answer for that.
    ///
    /// # Arguments
    ///
    /// * `method` - The method being rewritten.
    /// * `events` - Where to record what the pass did.
    /// * `context` - What the passes and their analyses run under: the
    ///   machine, image, budget, called functions' models and interrupt.
    #[must_use]
    pub const fn new(
        method: &'h T::MethodRef,
        events: &'h dyn EventListener<T>,
        context: AnalysisContext<'h>,
    ) -> Self {
        Self {
            events,
            ranges: RangeFacts::new(*context.budget()),
            context,
            binding: &(),
            method,
            world: None,
        }
    }

    /// Builds an environment that can also read the rest of the program.
    ///
    /// # Arguments
    ///
    /// * `method` - The method being rewritten.
    /// * `host` - The whole-program surface, also supplying events and what
    ///   the passes run under ([`SsaPassHost::context`]).
    #[must_use]
    pub fn from_host(method: &'h T::MethodRef, host: &'h dyn SsaPassHost<T>) -> Self {
        let context = host.context();
        Self {
            events: host.events(),
            ranges: RangeFacts::new(*context.budget()),
            context,
            binding: host.binding_oracle(),
            method,
            world: Some(host),
        }
    }

    /// Returns this environment with `image` as the image oracle.
    ///
    /// For a caller building an environment directly, whose host has an image
    /// to offer.
    #[must_use]
    pub const fn with_image(mut self, image: &'h dyn ImageData) -> Self {
        self.context = self.context.with_image(image);
        self
    }

    /// Returns this environment with `budget` as the work budget.
    ///
    /// The range facts start over: those held were computed within the old
    /// budget, and a different budget converges on different facts.
    #[must_use]
    pub fn with_budget(mut self, budget: WorkBudget) -> Self {
        self.context = self.context.with_budget(budget);
        self.ranges = RangeFacts::new(budget);
        self
    }

    /// Returns this environment with `interrupt` as what its pipeline asks,
    /// before each pass, whether it must stop
    /// ([`AnalysisContext::with_interrupt`]).
    #[must_use]
    pub const fn with_interrupt(mut self, interrupt: &'h dyn Interrupt) -> Self {
        self.context = self.context.with_interrupt(interrupt);
        self
    }

    /// Returns this environment with `oracle` as what the host knows about
    /// binding time.
    #[must_use]
    pub const fn with_binding_oracle(mut self, oracle: &'h dyn BindingOracle<T>) -> Self {
        self.binding = oracle;
        self
    }

    /// Returns what the host knows about binding time: `()`, knowing nothing,
    /// unless the host or the caller supplied an oracle.
    #[must_use]
    pub const fn binding_oracle(&self) -> &'h dyn BindingOracle<T> {
        self.binding
    }

    /// Returns the host's target facts.
    #[must_use]
    pub const fn context(&self) -> &AnalysisContext<'h> {
        &self.context
    }

    /// Returns where to record what the pass did.
    #[must_use]
    pub const fn events(&self) -> &'h dyn EventListener<T> {
        self.events
    }

    /// Returns the method being rewritten.
    #[must_use]
    pub const fn method(&self) -> &'h T::MethodRef {
        self.method
    }

    /// Opens an event of `kind` about the method being rewritten, appended to
    /// [`Self::events`] when the builder is dropped.
    ///
    /// The one way a pass records what it did: the method is filled in, and a
    /// listener that records nothing ([`EventListener::is_enabled`]) is handed
    /// no event.
    ///
    /// # Arguments
    ///
    /// * `kind` - What happened.
    pub fn record(&self, kind: EventKind) -> EventBuilder<'h, T, dyn EventListener<T> + 'h> {
        EventBuilder::new(self.events, kind).method(self.method.clone())
    }

    /// Records that a budget ran out while a pass rewrote the method, and the
    /// pass stopped with what it had ([`Event::exhausted`]).
    ///
    /// Pushed whatever [`EventListener::is_enabled`] says, as the scheduler
    /// pushes its own: a listener that builds no event may still count the
    /// budgets a function spent, and does so by returning `false` there and
    /// keeping the [`EventKind::BudgetExhausted`] events it is pushed.
    ///
    /// # Arguments
    ///
    /// * `exhaustion` - Which budget ran out, and by how much.
    pub fn exhausted(&self, exhaustion: Exhaustion) {
        self.events
            .push(Event::exhausted(Some(self.method), exhaustion));
    }

    /// Returns the rest of the program, when the caller had one to give.
    #[must_use]
    pub const fn world(&self) -> Option<&'h dyn SsaPassHost<T>> {
        self.world
    }

    /// Returns the method's range facts.
    ///
    /// Computed on first use and shared by every pass run under this
    /// environment until one changes the function: [`run_pass_with_repair`]
    /// invalidates them after a pass that did, and a driver that rolls a
    /// function back invalidates them then. A caller that changes the function
    /// any other way while it holds the environment calls
    /// [`RangeFacts::invalidate`] itself.
    #[must_use]
    pub fn ranges(&self) -> &RangeFacts {
        &self.ranges
    }

    /// Opens the analyses of `ssa`.
    ///
    /// Constructed against the function the caller is about to read, rather
    /// than held in the environment: an analysis borrows the function it
    /// describes, and a pass holds that function mutably. Reading the facts and
    /// then rewriting is the order that makes both possible - and the order a
    /// pass wants anyway, since a fact taken before a rewrite is the one the
    /// rewrite was decided on.
    ///
    /// Each slot is computed on first use and shared for the lifetime of the
    /// returned handle. Caching *across* passes is the host's
    /// [`AnalysisCache`](crate::scheduling::AnalysisCache), because only the
    /// host knows when a method changed.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to describe.
    #[must_use]
    pub fn analyses<'ir>(&self, ssa: &'ir SsaFunction<T>) -> FunctionAnalyses<'ir, T>
    where
        'h: 'ir,
    {
        FunctionAnalyses::new(ssa, &self.context)
    }
}

impl<T: Target> fmt::Debug for PassEnv<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PassEnv")
            .field("context", &self.context)
            .field("has_world", &self.world.is_some())
            .finish()
    }
}

/// A pass that rewrites one function, with no host type in its signature.
///
/// The shape almost every pass actually has. Writing one against
/// [`SsaPass`] instead means naming a host type it does not use, which is what
/// stopped a pass being shared between two host crates, or run by a caller that
/// has a function but no program.
///
/// A `FunctionPass` is an [`SsaPass`] wherever one is wanted: the blanket
/// bridge below adapts it, so a scheduler registers either kind.
pub trait FunctionPass<T: Target>: Send + Sync {
    /// Returns a unique short name, for logs and for pipeline reports.
    fn name(&self) -> &'static str;

    /// Returns a human-readable description.
    ///
    /// Purely informational; nothing branches on it.
    fn description(&self) -> &'static str {
        "No description available"
    }

    /// Returns what the pass may change, which decides the repair it needs.
    ///
    /// See [`ModificationScope`]. It has no default: a pass that changes the
    /// control-flow graph must repair its own SSA.
    fn modification_scope(&self) -> ModificationScope;

    /// Returns whether the pass leaves the SSA already repaired.
    ///
    /// A pass that runs its own reconstruction says so here, and the driver
    /// does not run a second one.
    fn repairs_ssa(&self) -> bool {
        false
    }

    /// Rewrites `ssa` in place.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite.
    /// * `env` - Events, pointer width, the method, and the program when there
    ///   is one.
    ///
    /// # Returns
    ///
    /// `Ok(true)` when the function changed.
    ///
    /// # Errors
    ///
    /// Returns `Err` when the pass cannot process the function.
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool>;
}

/// A pass that runs once over the whole program.
///
/// The counterpart to [`FunctionPass`] for work that is not about one function:
/// reachability over the call graph, and anything else whose question only has
/// an answer for the program as a whole.
///
/// Stated as its own trait because the alternative is what it replaces: a
/// whole-program pass declared as a per-function one, carrying a `run_on_method`
/// that is documented as never called and returns "nothing changed" to whoever
/// calls it anyway. That is a shape the type system could have ruled out, and
/// here it does - a `ModulePass` has no per-function entry point to leave
/// unimplemented.
pub trait ModulePass<T: Target>: Send + Sync {
    /// The pass's short name, as it appears in logs and reports.
    fn name(&self) -> &'static str;

    /// What the pass does, for a human reading a listing.
    fn description(&self) -> &'static str {
        ""
    }

    /// Runs the pass over the program.
    ///
    /// # Arguments
    ///
    /// * `world` - The program: its methods, entry points and call edges.
    /// * `events` - Where to record what the pass did.
    ///
    /// # Returns
    ///
    /// Whether anything changed.
    ///
    /// # Errors
    ///
    /// Returns `Err` when the pass cannot run over this program.
    fn run(&self, world: &dyn World<T>, events: &dyn EventListener<T>) -> Result<bool>;
}

/// A [`ModulePass`] where an [`SsaPass`] is wanted.
///
/// The same wrapping, and for the same reason, as [`AsSsaPass`]: a blanket impl
/// would collide with every concrete `SsaPass` a host has written.
#[derive(Debug, Clone, Copy, Default)]
pub struct AsGlobalPass<P>(pub P);

impl<T: Target, H: SsaPassHost<T>, P: ModulePass<T>> SsaPass<T, H> for AsGlobalPass<P> {
    fn name(&self) -> &'static str {
        ModulePass::name(&self.0)
    }

    /// Its per-method entry changes nothing; what the global run rewrites it
    /// repairs itself.
    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::UsesOnly
    }

    fn description(&self) -> &'static str {
        ModulePass::description(&self.0)
    }

    fn is_global(&self) -> bool {
        true
    }

    /// Never called: the pass is global, so the scheduler drives it through
    /// [`run_global`](SsaPass::run_global).
    fn run_on_method(
        &self,
        _ssa: &mut SsaFunction<T>,
        _method: &T::MethodRef,
        _host: &H,
    ) -> Result<bool> {
        Ok(false)
    }

    fn run_global(&self, host: &H) -> Result<bool> {
        ModulePass::run(&self.0, host, host.events())
    }
}

/// A [`FunctionPass`] where an [`SsaPass`] is wanted.
///
/// A named adapter rather than a blanket `impl<P: FunctionPass<T>> SsaPass<T,
/// H> for P`, and that is not a stylistic choice: a blanket impl would conflict
/// with *every* concrete `SsaPass` impl in existence, including the ones host
/// crates write for their own target-specific passes, because nothing can prove
/// those types do not also implement `FunctionPass`. Wrapping is explicit,
/// costs nothing at run time, and leaves a host's own passes exactly as they
/// were.
#[derive(Debug, Clone, Copy, Default)]
pub struct AsSsaPass<P>(pub P);

impl<T: Target, H: SsaPassHost<T>, P: FunctionPass<T>> SsaPass<T, H> for AsSsaPass<P> {
    fn name(&self) -> &'static str {
        FunctionPass::name(&self.0)
    }

    fn description(&self) -> &'static str {
        FunctionPass::description(&self.0)
    }

    fn modification_scope(&self) -> ModificationScope {
        FunctionPass::modification_scope(&self.0)
    }

    fn repairs_ssa(&self) -> bool {
        FunctionPass::repairs_ssa(&self.0)
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<T>,
        method: &T::MethodRef,
        host: &H,
    ) -> Result<bool> {
        FunctionPass::run(&self.0, ssa, &PassEnv::from_host(method, host))
    }
}

/// Runs one pass over one function, with the repair its scope calls for.
///
/// The scheduler does this around every pass it runs; this is the same
/// sequence for a caller running a pass outside one. Skipping it is a
/// specific, repeatable defect: a pass that changes control flow leaves stale
/// phis and def sites behind, and the next thing to read the function sees IR
/// that no longer describes itself.
///
/// # Arguments
///
/// * `pass` - The pass to run.
/// * `ssa` - The function to rewrite.
/// * `env` - The pass environment.
///
/// # Returns
///
/// Whether the function changed.
///
/// # Errors
///
/// Returns the pass's own error, and an error for a pass that changed the
/// control-flow graph without repairing the SSA itself: the values an edge
/// carries are the pass's to state ([`FunctionPass::repairs_ssa`]).
pub fn run_pass_with_repair<T: Target, P: FunctionPass<T> + ?Sized>(
    pass: &P,
    ssa: &mut SsaFunction<T>,
    env: &PassEnv<'_, T>,
) -> Result<bool> {
    let outcome = pass.run(ssa, env).and_then(|reported| {
        pass.modification_scope()
            .settle(ssa, pass.repairs_ssa(), reported)
            .map_err(|error| Error::new(format!("pass '{}' {error}", pass.name())))
    });
    // A pass that failed may have left edits behind, so only a run that
    // settled unchanged keeps the facts the next pass reads.
    if !matches!(outcome, Ok(false)) {
        env.ranges().invalidate();
    }
    outcome
}

/// An [`SsaPass`] with its capability declarations attached.
///
/// A pass written in this crate knows what it does but cannot name a value of
/// a host's capability enum, so it cannot say so. Until it can, the ordering
/// the scheduler computes rests on declarations each host repeats at every
/// registration - and a pass documented as providing a milestone provides it
/// only where somebody remembered.
///
/// This attaches the declaration to the pass instead, translated into the
/// host's own tags through [`Target::capability`]. A milestone the host does
/// not model is dropped rather than approximated: the pass then declares
/// nothing, which is what it truthfully provides there.
///
/// [`Target::capability`]: crate::target::Target::capability
pub struct Declared<T: Target, P> {
    /// The pass being declared for.
    pass: P,
    /// What it leaves true, in the host's tags.
    provides: Vec<T::Capability>,
    /// What must already be true, in the host's tags.
    requires: Vec<T::Capability>,
    /// Where it runs when capabilities do not constrain it.
    fallback_layer: usize,
}

impl<T: Target, P> Declared<T, P> {
    /// Wraps `pass` with no declarations.
    ///
    /// # Arguments
    ///
    /// * `pass` - The pass to declare for.
    #[must_use]
    pub fn new(pass: P) -> Self {
        Self {
            pass,
            provides: Vec::new(),
            requires: Vec::new(),
            fallback_layer: 0,
        }
    }

    /// Declares the milestones this pass leaves true.
    ///
    /// # Arguments
    ///
    /// * `shared` - The milestones, in the shared vocabulary.
    #[must_use]
    pub fn providing<I>(mut self, shared: I) -> Self
    where
        I: IntoIterator<Item = DeobfuscationCapability>,
    {
        self.provides
            .extend(shared.into_iter().filter_map(T::capability));
        self
    }

    /// Declares the milestones that must hold before this pass runs.
    ///
    /// # Arguments
    ///
    /// * `shared` - The milestones, in the shared vocabulary.
    #[must_use]
    pub fn requiring<I>(mut self, shared: I) -> Self
    where
        I: IntoIterator<Item = DeobfuscationCapability>,
    {
        self.requires
            .extend(shared.into_iter().filter_map(T::capability));
        self
    }

    /// Sets the layer this pass runs in when nothing else orders it.
    ///
    /// # Arguments
    ///
    /// * `layer` - The fallback layer.
    #[must_use]
    pub const fn in_layer(mut self, layer: usize) -> Self {
        self.fallback_layer = layer;
        self
    }
}

impl<T: Target, H: SsaPassHost<T>, P: SsaPass<T, H>> SsaPass<T, H> for Declared<T, P> {
    fn name(&self) -> &'static str {
        self.pass.name()
    }

    fn description(&self) -> &'static str {
        self.pass.description()
    }

    fn modification_scope(&self) -> ModificationScope {
        self.pass.modification_scope()
    }

    fn repairs_ssa(&self) -> bool {
        self.pass.repairs_ssa()
    }

    fn is_global(&self) -> bool {
        self.pass.is_global()
    }

    fn reads_peer_ssa(&self) -> bool {
        self.pass.reads_peer_ssa()
    }

    fn should_run(&self, method: &T::MethodRef, host: &H) -> bool {
        self.pass.should_run(method, host)
    }

    fn requires_full_scan(&self) -> bool {
        self.pass.requires_full_scan()
    }

    fn provides(&self) -> &[T::Capability] {
        &self.provides
    }

    fn requires(&self) -> &[T::Capability] {
        &self.requires
    }

    fn fallback_layer(&self) -> usize {
        self.fallback_layer
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<T>,
        method: &T::MethodRef,
        host: &H,
    ) -> Result<bool> {
        self.pass.run_on_method(ssa, method, host)
    }

    fn run_global(&self, host: &H) -> Result<bool> {
        self.pass.run_global(host)
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, process};

    use super::*;
    use crate::{
        analysis::{bindingtime::CallBinding, budget::Budget},
        events::{EventLog, NullListener},
        ir::{SsaVarId, instruction::SsaInstruction, ops::SsaOp},
        testing::{MACHINE, MockTarget, const_i32_return},
    };

    /// A pass that inserts a `Nop` ahead of everything else, which shifts
    /// every instruction index in the block and so invalidates the use lists -
    /// exactly the damage the repair exists to undo.
    struct InsertNop;

    impl FunctionPass<MockTarget> for InsertNop {
        fn name(&self) -> &'static str {
            "insert-nop"
        }

        fn modification_scope(&self) -> ModificationScope {
            ModificationScope::InstructionsOnly
        }

        fn run(
            &self,
            ssa: &mut SsaFunction<MockTarget>,
            _env: &PassEnv<'_, MockTarget>,
        ) -> Result<bool> {
            let Some(block) = ssa.block_mut(0) else {
                return Ok(false);
            };
            block
                .instructions_mut()
                .insert(0, SsaInstruction::synthetic(SsaOp::Nop));
            Ok(true)
        }
    }

    /// The same edit, declared as one that changes the control flow and
    /// leaving its repair to whoever runs it.
    struct InsertNopRestructuring;

    impl FunctionPass<MockTarget> for InsertNopRestructuring {
        fn name(&self) -> &'static str {
            "insert-nop-restructuring"
        }

        fn modification_scope(&self) -> ModificationScope {
            ModificationScope::CfgModifying
        }

        fn run(
            &self,
            ssa: &mut SsaFunction<MockTarget>,
            env: &PassEnv<'_, MockTarget>,
        ) -> Result<bool> {
            InsertNop.run(ssa, env)
        }
    }

    /// **A pass that changes the graph repairs its own SSA.** One that leaves
    /// the repair to whoever runs it is refused, since only the pass knows
    /// the values its edges carry; the same edit declared as keeping the graph
    /// is repaired after it.
    #[test]
    fn a_pass_leaving_a_graph_change_unrepaired_is_refused() {
        let env = PassEnv::<MockTarget>::new(&0u32, &NullListener, AnalysisContext::new(MACHINE));
        let mut ssa = const_i32_return(3);
        assert!(run_pass_with_repair(&InsertNopRestructuring, &mut ssa, &env).is_err());
        let mut ssa = const_i32_return(3);
        assert!(run_pass_with_repair(&InsertNop, &mut ssa, &env).is_ok());
    }

    #[test]
    fn a_function_pass_runs_without_a_host() {
        let mut ssa = const_i32_return(3);
        let env = PassEnv::<MockTarget>::new(&0u32, &NullListener, AnalysisContext::new(MACHINE));
        assert!(env.world().is_none());

        let changed =
            run_pass_with_repair(&InsertNop, &mut ssa, &env).unwrap_or_else(|_| process::abort());
        assert!(changed);
        // The repair ran. Without it every use site in the block names the
        // index it had before the insertion, which `validate` reports as a
        // stale use list and nothing else would notice until an analysis read
        // the wrong instruction.
        assert!(ssa.validate().is_ok(), "{ssa}");
    }

    #[test]
    fn a_pass_run_without_the_repair_leaves_the_function_describing_itself_wrongly() {
        // The control for the test above: the same edit, no repair.
        let mut ssa = const_i32_return(3);
        let env = PassEnv::<MockTarget>::new(&0u32, &NullListener, AnalysisContext::new(MACHINE));
        let changed = InsertNop
            .run(&mut ssa, &env)
            .unwrap_or_else(|_| process::abort());
        assert!(changed);
        assert!(
            ssa.validate().is_err(),
            "the use lists still describe the function as it was: {ssa}"
        );
    }

    /// **An environment built directly takes the host's binding oracle.**
    /// One knowing that every call computes of its arguments answers so
    /// once stated; the environment's own knows nothing.
    #[test]
    fn a_direct_environment_takes_a_binding_oracle() {
        #[derive(Debug)]
        struct Arguments;
        impl BindingOracle<MockTarget> for Arguments {
            fn call(&self, _method: &u32) -> CallBinding {
                CallBinding::OfArguments
            }
        }
        let plain = PassEnv::<MockTarget>::new(&0u32, &NullListener, AnalysisContext::new(MACHINE));
        assert_eq!(plain.binding_oracle().call(&1), CallBinding::Opaque);
        let stated = plain.with_binding_oracle(&Arguments);
        assert_eq!(stated.binding_oracle().call(&1), CallBinding::OfArguments);
    }

    #[test]
    fn the_environment_opens_the_analyses_of_the_function_at_hand() {
        let ssa = const_i32_return(1);
        let env = PassEnv::<MockTarget>::new(&0u32, &NullListener, AnalysisContext::new(MACHINE));
        let analyses = env.analyses(&ssa);
        assert_eq!(analyses.machine(), MACHINE);
        assert!(!analyses.def_use().is_absent(SsaVarId::from_index(0)));
    }

    /// A listener that builds no event and counts what it is handed anyway.
    #[derive(Default)]
    struct Disabled {
        pushed: Cell<usize>,
    }

    impl EventListener<MockTarget> for Disabled {
        fn push(&self, _event: Event<MockTarget>) {
            self.pushed.set(self.pushed.get() + 1);
        }

        fn is_enabled(&self) -> bool {
            false
        }
    }

    /// An event a pass records names the method the environment rewrites,
    /// with what the pass added.
    #[test]
    fn a_recorded_event_names_the_method() {
        let log = EventLog::<MockTarget>::new();
        let env = PassEnv::<MockTarget>::new(&7u32, &log, AnalysisContext::new(MACHINE));
        env.record(EventKind::ConstantFolded)
            .location(3)
            .message("folded");
        env.exhausted(Exhaustion {
            budget: Budget::RangeRounds,
            limit: 2,
            demand: 3,
        });
        let events: Vec<_> = log.iter().cloned().collect();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].kind, EventKind::ConstantFolded);
        assert_eq!(events[0].method, Some(7));
        assert_eq!(events[0].location, Some(3));
        assert_eq!(events[0].message, "folded");
        assert_eq!(events[0].pass, None);
        assert_eq!(events[1].kind, EventKind::BudgetExhausted);
        assert_eq!(events[1].method, Some(7));
        assert_eq!(
            events[1].message,
            Exhaustion {
                budget: Budget::RangeRounds,
                limit: 2,
                demand: 3,
            }
            .to_string()
        );
    }

    /// A listener that builds no event is handed none through the builder,
    /// and still every exhausted budget: a host counts those with such a
    /// listener.
    #[test]
    fn a_disabled_listener_is_handed_only_exhaustions() {
        let listener = Disabled::default();
        let env = PassEnv::<MockTarget>::new(&7u32, &listener, AnalysisContext::new(MACHINE));
        env.record(EventKind::ConstantFolded).message("folded");
        assert_eq!(listener.pushed.get(), 0);
        env.exhausted(Exhaustion {
            budget: Budget::RangeRounds,
            limit: 2,
            demand: 3,
        });
        assert_eq!(listener.pushed.get(), 1);
    }
}
