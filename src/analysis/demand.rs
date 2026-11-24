//! Observation dependencies for selecting useful execution contexts.
//!
//! Scalar, control, address and canonical memory requirements share the
//! binding-time producer. Unknown causes remain explicit so a host can bind
//! relevant caller inputs without requiring unrelated arguments to be static.

use std::{collections::BTreeSet, fmt, hash::Hash, mem, rc::Rc};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    analysis::{
        bindingtime::{BindingOracle, BindingTime, CallBinding},
        budget::Exhaustion,
        cache::FunctionAnalyses,
        cfg::SsaCfg,
        context::AnalysisContext,
        memory::{ArrayIndex, MemoryDefSite, MemoryLocation, MemoryOp, MemorySsa},
        pointsto::analyze_function,
    },
    bitset::BitSet,
    graph::{
        NodeId, RootedGraph,
        algorithms::{DominatorTree, ReverseGraph, compute_dominators},
    },
    ir::{
        function::SsaFunction,
        ops::{CallData, SsaOp, effects::SsaEffectKind},
        variable::SsaVarId,
    },
    target::Target,
};

/// The information an input supplies to an observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DependencyRole {
    /// A computed scalar or stored value.
    Value,
    /// An object identity, pointer or index selecting accessed bytes.
    Address,
    /// A condition selecting incoming state or whether the observation executes.
    Control,
    /// A requested extent, such as the length of an observed buffer.
    Bound,
}

/// An observation in the immutable body borrowed by [`Dependencies`].
///
/// Positions belong to that body revision. A host persisting a request must
/// additionally retain its body identity and revalidate positions after edits.
#[derive(Debug, Clone, Copy)]
pub enum Observation<'a, T: Target> {
    /// One scalar, with its role in the requested result.
    Value(SsaVarId, DependencyRole),
    /// One canonical memory cell strictly before an instruction.
    ///
    /// The block length denotes normal exit. Only locations classified by this
    /// analysis are accepted; this does not manufacture an arbitrary byte span.
    MemoryBefore {
        /// The cell whose contents are requested.
        location: &'a MemoryLocation<T>,
        /// The block containing the observation.
        block: usize,
        /// The position strictly after the requested state.
        instruction: usize,
    },
    /// The conditions selecting a block's incoming state and whether it executes.
    Control(usize),
}

/// The source of contents at an observed memory version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryOrigin {
    /// Unknown contents supplied before this invocation.
    Entry,
    /// Lifetime-immutable image contents; the image remains a dependency.
    Image,
    /// Contents whose binding time is supplied by a host entry contract.
    Host,
    /// A possible write or call defining this local version.
    Write {
        /// The defining block.
        block: usize,
        /// The defining instruction.
        instruction: usize,
    },
    /// A merge of ordered predecessor states.
    Merge {
        /// The merge block.
        block: usize,
    },
    /// State supplied by exceptional entry.
    Exception {
        /// The handler/filter block.
        block: usize,
    },
}

/// One input, assumption or unresolved effect retained by a request.
///
/// These are dependencies, not evaluated facts. A static pointer does not
/// resolve a memory dependency, and an empty list does not prove termination,
/// absence of traps or permission to replace an operation.
#[derive(Debug)]
pub enum Dependency<T: Target> {
    /// A caller-supplied SSA value, retaining the role it must supply.
    Input {
        /// The value to substitute through the host's actual ABI binding.
        value: SsaVarId,
        /// What the observation requires of the value.
        role: DependencyRole,
    },
    /// A canonical cell's local version and origin, independently of its address.
    ///
    /// A possible or partial writer retains the preceding version too. This
    /// dependency does not promise that a write executed or replaced every byte.
    Memory {
        /// The canonical cell in this analysis, held once for every dependency
        /// naming it.
        location: Rc<MemoryLocation<T>>,
        /// Analysis-local state identity, never a cross-context cache key.
        version: u32,
        /// What supplies the contents.
        origin: MemoryOrigin,
    },
    /// An unresolved operation result or may-alias memory writer.
    Effect {
        /// The containing block.
        block: usize,
        /// The effect's instruction position.
        instruction: usize,
    },
    /// Whether an exception transfers control or supplies unknown state.
    Exception {
        /// The affected or deciding block.
        block: usize,
    },
    /// A host binding contract used for a call result or filled object.
    ///
    /// This retains the assumption; it does not assert that the call is pure.
    CallContract {
        /// The call's block.
        block: usize,
        /// The call's instruction position.
        instruction: usize,
    },
    /// A value or control/memory relation the analysis cannot identify.
    Unknown {
        /// The value, when a malformed or unsupported definition names one.
        value: Option<SsaVarId>,
    },
}

impl<T: Target> Clone for Dependency<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Input { value, role } => Self::Input {
                value: *value,
                role: *role,
            },
            Self::Memory {
                location,
                version,
                origin,
            } => Self::Memory {
                location: Rc::clone(location),
                version: *version,
                origin: *origin,
            },
            Self::Effect { block, instruction } => Self::Effect {
                block: *block,
                instruction: *instruction,
            },
            Self::Exception { block } => Self::Exception { block: *block },
            Self::CallContract { block, instruction } => Self::CallContract {
                block: *block,
                instruction: *instruction,
            },
            Self::Unknown { value } => Self::Unknown { value: *value },
        }
    }
}

/// The dependencies one body's requests have named, each held once, by its
/// number: the order it was first met in.
///
/// It only grows, by a segment a request adds, so a signature taken before a
/// later request still reads every number it holds.
struct DependencyTable<T: Target> {
    /// The dependencies, in segments as requests added them.
    segments: Vec<Rc<[Dependency<T>]>>,
    /// The number of each segment's first dependency.
    starts: Vec<usize>,
    /// How many there are.
    len: usize,
}

impl<T: Target> Clone for DependencyTable<T> {
    fn clone(&self) -> Self {
        Self {
            segments: self.segments.clone(),
            starts: self.starts.clone(),
            len: self.len,
        }
    }
}

impl<T: Target> Default for DependencyTable<T> {
    fn default() -> Self {
        Self {
            segments: Vec::new(),
            starts: Vec::new(),
            len: 0,
        }
    }
}

impl<T: Target> fmt::Debug for DependencyTable<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.segments.iter().flat_map(|segment| segment.iter()))
            .finish()
    }
}

impl<T: Target> DependencyTable<T> {
    /// The dependency numbered `number`, if the table holds one.
    #[must_use]
    pub fn get(&self, number: usize) -> Option<&Dependency<T>> {
        let segment = self
            .starts
            .partition_point(|start| *start <= number)
            .checked_sub(1)?;
        let start = *self.starts.get(segment)?;
        self.segments.get(segment)?.get(number.checked_sub(start)?)
    }

    /// How many dependencies the table holds.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Adds `dependencies`, numbered on from the last held.
    ///
    /// A table a signature still shares is copied to grow, segment list and
    /// all, so a segment no larger than the one added is merged into it: a
    /// table holds a logarithmic number of segments, and a body's requests
    /// copy a few each rather than one for every request before them.
    fn extend(&mut self, dependencies: Vec<Dependency<T>>) {
        if dependencies.is_empty() {
            return;
        }
        let added = dependencies.len();
        let mut segment = dependencies;
        let mut start = self.len;
        while self
            .segments
            .last()
            .is_some_and(|last| last.len() <= segment.len())
        {
            let (Some(last), Some(its)) = (self.segments.pop(), self.starts.pop()) else {
                break;
            };
            let mut merged = Vec::with_capacity(last.len().saturating_add(segment.len()));
            merged.extend(last.iter().cloned());
            merged.append(&mut segment);
            segment = merged;
            start = its;
        }
        self.starts.push(start);
        self.len = self.len.saturating_add(added);
        self.segments.push(Rc::from(segment));
    }
}

/// A request's relevant inputs and effects: those of its body's dependencies it
/// names, in the order the body first met them, and any a host stated beside
/// them ([`Self::with`]).
///
/// The requests of one body name mostly the same dependencies, often nearly
/// all of them; each signature names them by number in one table the body's
/// requests share, rather than holding a copy of each.
pub struct DependencySignature<T: Target> {
    /// The body's dependencies.
    table: Rc<DependencyTable<T>>,
    /// The numbers of those the request names.
    members: Rc<BitSet>,
    /// Dependencies a host stated beside the request's own.
    stated: Vec<Dependency<T>>,
}

impl<T: Target> Clone for DependencySignature<T> {
    fn clone(&self) -> Self {
        Self {
            table: Rc::clone(&self.table),
            members: Rc::clone(&self.members),
            stated: self.stated.clone(),
        }
    }
}

impl<T: Target> fmt::Debug for DependencySignature<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.dependencies()).finish()
    }
}

impl<T: Target> DependencySignature<T> {
    /// A signature of exactly `dependencies`, in order, in a table of their
    /// own.
    #[must_use]
    pub fn of(dependencies: Vec<Dependency<T>>) -> Self {
        let mut members = BitSet::new(dependencies.len());
        for number in 0..dependencies.len() {
            members.insert(number);
        }
        let mut table = DependencyTable::default();
        table.extend(dependencies);
        Self {
            table: Rc::new(table),
            members: Rc::new(members),
            stated: Vec::new(),
        }
    }

    /// The dependencies the request names, each once, in the order its body
    /// first met them, then those stated beside it.
    pub fn dependencies(&self) -> impl Iterator<Item = &Dependency<T>> {
        self.members
            .iter()
            .filter_map(|number| self.table.get(number))
            .chain(self.stated.iter())
    }

    /// How many dependencies the signature names.
    #[must_use]
    pub fn len(&self) -> usize {
        self.members.count().saturating_add(self.stated.len())
    }

    /// Whether the signature names none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The signature with `dependency` stated beside the request's own: what a
    /// host knows the observation also depends on, which no definition the
    /// request traced names.
    #[must_use]
    pub fn with(mut self, dependency: Dependency<T>) -> Self {
        self.stated.push(dependency);
        self
    }
}

/// Why no complete dependency signature was published.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependencyRefusal {
    /// A position, cell or value does not belong to the borrowed body.
    InvalidObservation,
    /// Canonical memory analysis refused its retained-cell allowance.
    Memory(Exhaustion),
}

/// A compact edge shared by binding-time and observation requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Input {
    /// The value supplying this edge.
    value: SsaVarId,
    /// Its semantic role in the dependency graph.
    role: DependencyRole,
    /// Whether this edge also determines legacy scalar binding time.
    binding: bool,
}

impl Input {
    /// Names an ordinary scalar dependency.
    fn value(value: SsaVarId) -> Self {
        Self {
            value,
            role: DependencyRole::Value,
            binding: true,
        }
    }
}

/// An identity a request reaches, numbered once per body in its
/// [`DependencyTable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Cause {
    /// A caller value and its use role.
    Input(SsaVarId, DependencyRole),
    /// An interned canonical cell, local version and contents origin.
    Memory(usize, u32, MemoryOrigin),
    /// An operation whose output/effect is not resolved.
    Effect(usize, usize),
    /// Unknown exceptional state or transfer.
    Exception(usize),
    /// A host call binding assumption.
    Call(usize, usize),
    /// An unidentified relation.
    Unknown(Option<SsaVarId>),
}

/// What one value's binding time is computed from.
#[derive(Debug, Clone, Default)]
struct Source {
    /// Dynamic whatever the values below are.
    dynamic: bool,
    /// The values whose join it is. None at all is static.
    values: Vec<Input>,
    /// Unresolved state and host assumptions, without losing their identities.
    causes: Vec<Cause>,
    /// Whether it is a location's contents rather than a value a store handed
    /// over, and so depends on which cell the load selects.
    contents: bool,
}

impl Source {
    /// A source that is dynamic outright.
    fn dynamic() -> Self {
        Self {
            dynamic: true,
            ..Self::default()
        }
    }

    /// Derives ordinary value edges.
    fn of(values: impl IntoIterator<Item = SsaVarId>) -> Self {
        let mut source = Self::default();
        source.values.extend(values.into_iter().map(Input::value));
        source
    }

    /// Adds the dynamic index selecting a loaded cell's contents.
    fn add_index_of<T: Target>(&mut self, location: &MemoryLocation<T>) {
        let index = match location {
            MemoryLocation::ArrayElement(_, ArrayIndex::Variable(index)) => Some(*index),
            MemoryLocation::Indirect(cell) => cell.index,
            MemoryLocation::ArrayElement(_, ArrayIndex::Unknown)
            | MemoryLocation::Unknown
            | MemoryLocation::Escaped => {
                self.dynamic = true;
                None
            }
            _ => None,
        };
        if let Some(value) = index {
            self.values.push(Input {
                value,
                role: DependencyRole::Address,
                binding: true,
            });
        }
    }

    /// Retains the object and index selecting a written or observed memory cell.
    fn add_address_of<T: Target>(&mut self, location: &MemoryLocation<T>, binding: bool) {
        let index = match location {
            MemoryLocation::Indirect(cell) => cell.index,
            MemoryLocation::ArrayElement(_, ArrayIndex::Variable(index)) => Some(*index),
            _ => None,
        };
        self.values.extend(
            [location.base_object(), index]
                .into_iter()
                .flatten()
                .map(|value| Input {
                    value,
                    role: DependencyRole::Address,
                    binding,
                }),
        )
    }

    /// Adds a binding time supplied independently of scalar definitions.
    fn add_time(&mut self, time: BindingTime) {
        self.dynamic |= time == BindingTime::Dynamic;
    }

    /// Joins another source's edges and identities into this one, each held
    /// once.
    ///
    /// A source is a set: what reads it asks whether an edge or an identity
    /// is there, never how often. A load reached by many stores joins the
    /// conditions each store's block is reached under, and stores of one body
    /// share most of them, so appending would hold every shared condition
    /// once per store and every walk over the source would read them again.
    fn absorb(&mut self, other: &Self) {
        self.dynamic |= other.dynamic;
        self.contents |= other.contents;
        Self::extend_distinct(&mut self.values, &other.values);
        Self::extend_distinct(&mut self.causes, &other.causes);
    }

    /// This source with each edge and identity held once, the first of each
    /// in place, for a source built by pushes a walk may repeat.
    fn distinct(mut self) -> Self {
        let mut values = FxHashSet::default();
        self.values.retain(|input| values.insert(*input));
        let mut causes = FxHashSet::default();
        self.causes.retain(|cause| causes.insert(*cause));
        self
    }

    /// Appends the members of `from` that `into` does not hold yet, in order.
    fn extend_distinct<X: Copy + Eq + Hash>(into: &mut Vec<X>, from: &[X]) {
        if from.is_empty() {
            return;
        }
        let mut held: FxHashSet<X> = into.iter().copied().collect();
        into.extend(from.iter().copied().filter(|member| held.insert(*member)));
    }
}

/// Every defined value's [`Source`], and the reverse edges the solve walks.
struct Solve {
    /// The source of each defined value.
    sources: Vec<(SsaVarId, Source)>,
}

impl Solve {
    /// Solves the least fixpoint: every value static that nothing makes dynamic.
    ///
    /// A value only ever moves from static to dynamic, so the solve is a
    /// forward walk from the dynamic ones along the edges from a value to what
    /// is computed from it.
    ///
    /// # Arguments
    ///
    /// * `bound` - One past the largest value id.
    fn solve(self, bound: usize) -> Vec<BindingTime> {
        let mut times = vec![BindingTime::Dynamic; bound];
        let mut readers: FxHashMap<SsaVarId, Vec<SsaVarId>> = FxHashMap::default();
        for (value, _) in &self.sources {
            if let Some(time) = times.get_mut(value.index()) {
                *time = BindingTime::Static;
            }
        }
        let mut dynamic: Vec<SsaVarId> = Vec::new();
        for (value, source) in &self.sources {
            if source.dynamic {
                dynamic.push(*value);
                continue;
            }
            for input in &source.values {
                if input.binding {
                    readers.entry(input.value).or_default().push(*value);
                }
            }
        }
        // Undefined values are dynamic from the start, and seed the walk too.
        dynamic.extend(
            readers
                .keys()
                .filter(|value| {
                    times
                        .get(value.index())
                        .is_none_or(|time| *time == BindingTime::Dynamic)
                })
                .copied(),
        );
        for value in &dynamic {
            if let Some(time) = times.get_mut(value.index()) {
                *time = BindingTime::Dynamic;
            }
        }
        while let Some(value) = dynamic.pop() {
            for reader in readers.get(&value).map_or(&[][..], Vec::as_slice) {
                if let Some(time) = times.get_mut(reader.index())
                    && *time == BindingTime::Static
                {
                    *time = BindingTime::Dynamic;
                    dynamic.push(*reader);
                }
            }
        }
        times
    }
}

/// Dependency requests over one immutable function and its canonical memory.
///
/// Source derivation is shared with [`crate::analysis::bindingtime::BindingTimes`].
/// Requests retain unresolved identities even when one dependency is dynamic.
/// They select evaluation inputs; they do not compute values or prove rewrites.
/// Construct this only for selected candidates, never as whole-image screening.
pub struct Dependencies<'a, T: Target> {
    /// The function.
    ssa: &'a SsaFunction<T>,
    /// Its Memory SSA.
    memory: MemorySsa<T>,
    /// The context the Memory SSA was built in, which places a location in its
    /// region.
    context: &'a AnalysisContext<'a>,
    /// What the host knows about calls and initial contents.
    oracle: &'a dyn BindingOracle<T>,
    /// The exception-aware control flow graph.
    cfg: SsaCfg<'a, T>,
    /// Dominance over it.
    dominators: DominatorTree,
    /// Lazy definition/use queries tied to the same immutable body.
    analyses: FunctionAnalyses<'a, T>,
    /// Whether unresolved identities and address-only edges are requested.
    tracing: bool,
    /// Derived value sources, retained across requests for this body.
    sources: FxHashMap<SsaVarId, Source>,
    /// The branches deciding each merge block's incoming edge, once computed.
    control: FxHashMap<usize, Source>,
    /// Sparse direct controllers, derived lazily from canonical post-dominance.
    controllers: Option<Vec<Vec<usize>>>,
    /// Transitive conditions deciding whether an instruction block is reached.
    reach: FxHashMap<usize, Source>,
    /// Every dependency a request has named, by number, shared with the
    /// signatures that name them.
    table: Rc<DependencyTable<T>>,
    /// Each identity's number in [`Self::table`].
    numbers: FxHashMap<Cause, usize>,
    /// Dependencies numbered since the table was last extended.
    fresh: Vec<Dependency<T>>,
    /// Each canonical cell named by a dependency, held once.
    locations: FxHashMap<usize, Rc<MemoryLocation<T>>>,
    /// The dependencies each value reaches in each role, by number: one set
    /// for every value of a cycle ([`Self::closure`]).
    closures: FxHashMap<Node, Rc<BitSet>>,
}

/// A value and the role it is read in: what reaches a dependency, and what a
/// closure is held for ([`Dependencies::closure`]).
type Node = (SsaVarId, DependencyRole);

impl<'a, T: Target> Dependencies<'a, T> {
    /// Prepares canonical memory and control analyses for selected-body requests.
    ///
    /// This is candidate analysis, not a cheap inventory operation. MemorySSA,
    /// CFG and dominance construction happen here.
    #[must_use]
    pub fn analyze(
        ssa: &'a SsaFunction<T>,
        context: &'a AnalysisContext<'a>,
        oracle: &'a dyn BindingOracle<T>,
    ) -> Self {
        let cfg = SsaCfg::from_ssa(ssa);
        let dominators = compute_dominators(&cfg, cfg.entry());
        let memory = MemorySsa::build_with_cfg(&cfg, context, Some(&dominators), || {
            analyze_function(ssa, context.budget(), context.models())
        });
        Self {
            ssa,
            memory,
            context,
            oracle,
            cfg,
            dominators,
            analyses: FunctionAnalyses::new(ssa, context),
            tracing: true,
            sources: FxHashMap::default(),
            control: FxHashMap::default(),
            controllers: None,
            reach: FxHashMap::default(),
            table: Rc::new(DependencyTable::default()),
            numbers: FxHashMap::default(),
            fresh: Vec::new(),
            locations: FxHashMap::default(),
            closures: FxHashMap::default(),
        }
    }

    /// Returns the retained-cell refusal of the canonical memory analysis.
    #[must_use]
    pub const fn exhaustion(&self) -> Option<Exhaustion> {
        self.memory.exhaustion()
    }

    /// Computes scalar binding times from the same dependency producer.
    pub fn binding_times(mut self) -> Vec<BindingTime> {
        self.tracing = false;
        let bound = self.ssa.var_id_bound();
        self.collect().solve(bound)
    }

    /// Traces one observation through relevant definitions, memory and control.
    ///
    /// The signature is owned, and shares this body's dependencies with every
    /// other signature of it. Its SSA ids and memory versions are local to this
    /// body and analysis: a host keeping it keeps the body revision with it.
    ///
    /// # Errors
    ///
    /// Returns invalid observations or canonical memory exhaustion without
    /// publishing a partial signature.
    pub fn request(
        &mut self,
        observation: Observation<'_, T>,
    ) -> Result<DependencySignature<T>, DependencyRefusal> {
        if let Some(exhaustion) = self.memory.exhaustion() {
            return Err(DependencyRefusal::Memory(exhaustion));
        }
        let root = self.observation_source(observation)?;
        self.trace(root)
    }

    /// Traces a complete observation from scalar, address, memory and control roots together.
    ///
    /// Repeated value/role pairs and dependency identities are named once for
    /// the group. Keeping a store's address and data together prevents a fixed
    /// address from appearing to satisfy an otherwise unknown stored value.
    /// An empty group describes no output and establishes no execution fact.
    ///
    /// # Errors
    ///
    /// Returns an invalid observation or canonical memory exhaustion without a
    /// partial signature when any member of the group is refused.
    pub fn request_all(
        &mut self,
        observations: &[Observation<'_, T>],
    ) -> Result<DependencySignature<T>, DependencyRefusal> {
        if let Some(exhaustion) = self.memory.exhaustion() {
            return Err(DependencyRefusal::Memory(exhaustion));
        }
        if let [observation] = observations {
            return self.request(observation.clone());
        }
        let mut root = Source::default();
        for observation in observations {
            if let Observation::Value(value, role) = observation {
                if self.ssa.variable(*value).is_none() {
                    return Err(DependencyRefusal::InvalidObservation);
                }
                root.values.push(Input {
                    value: *value,
                    role: *role,
                    binding: true,
                });
            } else {
                root.absorb(&self.observation_source(observation.clone())?);
            }
        }
        self.trace(root)
    }

    /// Resolves one requested root without publishing a signature.
    fn observation_source(
        &mut self,
        observation: Observation<'_, T>,
    ) -> Result<Source, DependencyRefusal> {
        let root = match observation {
            Observation::Value(value, role) => {
                if self.ssa.variable(value).is_none() {
                    return Err(DependencyRefusal::InvalidObservation);
                }
                let mut source = Source::default();
                source.values.push(Input {
                    value,
                    role,
                    binding: true,
                });
                source
            }
            Observation::MemoryBefore {
                location,
                block,
                instruction,
            } => {
                if let Some(exhaustion) = self.memory.exhaustion() {
                    return Err(DependencyRefusal::Memory(exhaustion));
                }
                let Some(id) = self.memory.location_id(location) else {
                    return Err(DependencyRefusal::InvalidObservation);
                };
                if self
                    .memory
                    .version_before_id(id, block, instruction)
                    .is_none()
                {
                    return Err(DependencyRefusal::InvalidObservation);
                }
                let mut source = self.reaching(block, instruction, id);
                source.absorb(self.reach_control(block));
                source
            }
            Observation::Control(block) => {
                if self.ssa.block(block).is_none() {
                    return Err(DependencyRefusal::InvalidObservation);
                }
                let mut source = self.control(block).clone();
                source.absorb(self.reach_control(block));
                source
            }
        };
        Ok(root)
    }

    /// The signature of `root`: its own identities, read as values, and
    /// everything its inputs reach ([`Self::closure`]), each dependency once.
    fn trace(&mut self, root: Source) -> Result<DependencySignature<T>, DependencyRefusal> {
        let mut members: Option<Rc<BitSet>> = None;
        let mut own = BitSet::new(0);
        for cause in &root.causes {
            let cause = match cause {
                Cause::Input(value, _) => Cause::Input(*value, DependencyRole::Value),
                other => *other,
            };
            let number = self.number(cause)?;
            own.ensure_capacity(number);
            own.insert(number);
        }
        let mut union: Option<BitSet> = (!own.is_empty()).then_some(own);
        for input in &root.values {
            let closure = self.closure((input.value, input.role))?;
            match (&mut union, &members) {
                (Some(union), _) => {
                    union.union_growing(&closure);
                }
                // The first input's closure, shared while it is the only one.
                (None, None) => members = Some(closure),
                (None, Some(first)) => {
                    let mut joined = BitSet::clone(first);
                    joined.union_growing(&closure);
                    union = Some(joined);
                }
            }
        }
        let members = match union {
            Some(union) => Rc::new(union),
            None => members.unwrap_or_else(|| Rc::new(BitSet::new(0))),
        };
        if !self.fresh.is_empty() {
            Rc::make_mut(&mut self.table).extend(mem::take(&mut self.fresh));
        }
        Ok(DependencySignature {
            table: Rc::clone(&self.table),
            members,
            stated: Vec::new(),
        })
    }

    /// The number of `cause` in [`Self::table`], numbering it if it is new.
    fn number(&mut self, cause: Cause) -> Result<usize, DependencyRefusal> {
        if let Some(number) = self.numbers.get(&cause) {
            return Ok(*number);
        }
        let dependency = match cause {
            Cause::Input(value, role) => Dependency::Input { value, role },
            Cause::Memory(id, version, origin) => {
                let location = match self.locations.get(&id) {
                    Some(location) => Rc::clone(location),
                    None => {
                        let location = Rc::new(
                            self.memory
                                .location_by_id(id)
                                .ok_or(DependencyRefusal::InvalidObservation)?
                                .clone(),
                        );
                        self.locations.insert(id, Rc::clone(&location));
                        location
                    }
                };
                Dependency::Memory {
                    location,
                    version,
                    origin,
                }
            }
            Cause::Effect(block, instruction) => Dependency::Effect { block, instruction },
            Cause::Exception(block) => Dependency::Exception { block },
            Cause::Call(block, instruction) => Dependency::CallContract { block, instruction },
            Cause::Unknown(value) => Dependency::Unknown { value },
        };
        let number = self.table.len().saturating_add(self.fresh.len());
        self.fresh.push(dependency);
        self.numbers.insert(cause, number);
        Ok(number)
    }

    /// What `node` reads, and its own identities in its role, by number: its
    /// value's source, each input read in the role it names, or in `node`'s
    /// own where it names a value.
    fn expand(&mut self, (value, role): Node) -> Result<(Vec<Node>, BitSet), DependencyRefusal> {
        if !self.sources.contains_key(&value) {
            let source = self.value_source(value).distinct();
            self.sources.insert(value, source);
        }
        let (inputs, causes) = self
            .sources
            .get(&value)
            .map(|source| (source.values.clone(), source.causes.clone()))
            .unwrap_or_default();
        let mut own = BitSet::new(0);
        for cause in causes {
            let cause = match cause {
                Cause::Input(value, _) => Cause::Input(value, role),
                other => other,
            };
            let number = self.number(cause)?;
            own.ensure_capacity(number);
            own.insert(number);
        }
        let reads = inputs
            .iter()
            .map(|input| {
                let read = if input.role == DependencyRole::Value {
                    role
                } else {
                    input.role
                };
                (input.value, read)
            })
            .collect();
        Ok((reads, own))
    }

    /// The dependencies `root` reaches, by number: its own identities and
    /// everything its inputs reach.
    ///
    /// Held once a value and role are asked for. What a value reaches is a
    /// graph that cycles where a loop carries a value round, and every value of
    /// one cycle reaches the same dependencies: the graph's strongly connected
    /// components are found as it is walked (Tarjan's, on a stack of its own,
    /// whatever its depth), and each component is given one set, its values'
    /// own identities and the sets of the components it reads. A body's
    /// requests then share what they reach instead of walking it again each:
    /// its every call site may depend on nearly all of it.
    fn closure(&mut self, root: Node) -> Result<Rc<BitSet>, DependencyRefusal> {
        if let Some(known) = self.closures.get(&root) {
            return Ok(Rc::clone(known));
        }
        // Each node's place in the walk and the least place it reaches back to.
        let mut places: FxHashMap<Node, (usize, usize)> = FxHashMap::default();
        let mut own: FxHashMap<Node, BitSet> = FxHashMap::default();
        let mut reads: FxHashMap<Node, Vec<Node>> = FxHashMap::default();
        let mut component: Vec<Node> = Vec::new();
        let mut open: FxHashSet<Node> = FxHashSet::default();
        // The walk: each node with the reads it has yet to follow.
        let mut walk: Vec<(Node, usize)> = Vec::new();
        let (first, mine) = self.expand(root)?;
        places.insert(root, (0, 0));
        own.insert(root, mine);
        reads.insert(root, first);
        component.push(root);
        open.insert(root);
        walk.push((root, 0));
        while let Some(&(node, next)) = walk.last() {
            let read = reads.get(&node).and_then(|all| all.get(next)).copied();
            if let Some(read) = read {
                if let Some(top) = walk.last_mut() {
                    top.1 = next.saturating_add(1);
                }
                if self.closures.contains_key(&read) {
                    continue;
                }
                match places.get(&read) {
                    None => {
                        let place = places.len();
                        let (its, mine) = self.expand(read)?;
                        places.insert(read, (place, place));
                        own.insert(read, mine);
                        reads.insert(read, its);
                        component.push(read);
                        open.insert(read);
                        walk.push((read, 0));
                    }
                    Some(&(place, _)) if open.contains(&read) => {
                        if let Some(entry) = places.get_mut(&node) {
                            entry.1 = entry.1.min(place);
                        }
                    }
                    Some(_) => {}
                }
                continue;
            }
            walk.pop();
            let (place, low) = places.get(&node).copied().unwrap_or((0, 0));
            if let Some(&(parent, _)) = walk.last()
                && let Some(entry) = places.get_mut(&parent)
            {
                entry.1 = entry.1.min(low);
            }
            if low != place {
                continue;
            }
            // `node` heads a component: it and every node above it on the
            // stack reach one another, and share one set.
            let mut members = Vec::new();
            while let Some(member) = component.pop() {
                open.remove(&member);
                members.push(member);
                if member == node {
                    break;
                }
            }
            let mut closure = BitSet::new(0);
            for member in &members {
                if let Some(mine) = own.get(member) {
                    closure.union_growing(mine);
                }
                for read in reads.get(member).into_iter().flatten() {
                    if let Some(theirs) = self.closures.get(read) {
                        closure.union_growing(theirs);
                    }
                }
            }
            let closure = Rc::new(closure);
            for member in members {
                self.closures.insert(member, Rc::clone(&closure));
            }
        }
        self.closures
            .get(&root)
            .cloned()
            .ok_or(DependencyRefusal::InvalidObservation)
    }

    /// Derives one defined value or retains its unresolved entry identity.
    fn value_source(&mut self, value: SsaVarId) -> Source {
        let site = self.analyses.def_use().active_def_site(value);
        if let Some(site) = site {
            if let Some(instruction) = site.index() {
                let mut source = self.instruction_source(site.block(), instruction);
                source.absorb(self.reach_control(site.block()));
                return source;
            }
            if let Some(block) = self.ssa.block(site.block()) {
                for phi in block.phi_nodes() {
                    if phi.result() == value {
                        let mut source = Source::of(phi.used_variables());
                        source.absorb(self.control(site.block()));
                        source.absorb(self.reach_control(site.block()));
                        return source;
                    }
                }
            }
        }
        let mut source = Source::dynamic();
        self.remember(
            &mut source,
            if self.ssa.variable(value).is_some_and(|variable| {
                variable.origin().is_caller_supplied() && variable.version() == 0
            }) {
                Cause::Input(value, DependencyRole::Value)
            } else {
                Cause::Unknown(Some(value))
            },
        );
        source
    }

    /// Retains identities for observation requests; scalar binding needs no trace.
    fn remember(&self, source: &mut Source, cause: Cause) {
        if self.tracing {
            source.causes.push(cause);
        }
    }

    /// Derives every defined value for ordinary, infallible scalar binding analysis.
    fn collect(mut self) -> Solve {
        let mut sources = Vec::new();
        for (block_idx, block) in self.ssa.blocks().iter().enumerate() {
            for phi in block.phi_nodes() {
                let mut source = self.control(block_idx).clone();
                let operands = Source::of(phi.used_variables());
                source.absorb(&operands);
                sources.push((phi.result(), source));
            }
            for (instr_idx, instruction) in block.instructions().iter().enumerate() {
                let mut defs = instruction.defs().peekable();
                if defs.peek().is_none() {
                    continue;
                }
                let source = self.instruction_source(block_idx, instr_idx).distinct();
                for def in defs {
                    sources.push((def, source.clone()));
                }
            }
        }
        Solve { sources }
    }

    /// Derives result requirements without allocating an operand list.
    fn instruction_source(&mut self, block: usize, instruction: usize) -> Source {
        if let Some(MemoryOp::Load { location, .. }) = self.memory.operation_at(block, instruction)
            && let Some(id) = self.memory.location_id(location)
        {
            return self.reaching(block, instruction, id);
        }
        let Some(op) = self
            .ssa
            .block(block)
            .and_then(|b| b.instructions().get(instruction))
            .map(|i| i.op())
        else {
            let mut source = Source::dynamic();
            self.remember(&mut source, Cause::Unknown(None));
            return source;
        };
        if let Some((CallBinding::OfArguments, call)) = self.call(op) {
            let mut source = Source::of(
                call.args
                    .iter()
                    .copied()
                    .chain(call.implicit_inputs.iter().map(|input| input.value)),
            );
            self.remember(&mut source, Cause::Call(block, instruction));
            return source;
        }
        if !self.tracing && op.effects().kind != SsaEffectKind::Pure {
            return Source::dynamic();
        }
        let mut source = Source::default();
        op.for_each_use(|value| source.values.push(Input::value(value)));
        if self.tracing {
            // Canonical computed/atomic reads need their reaching state even
            // when the result cannot be forwarded from one stored scalar.
            let mut locations = Vec::new();
            if let Some(operation) = self.memory.operation_at(block, instruction) {
                for location in operation.read_locations() {
                    if let Some(id) = self.memory.location_id(location) {
                        locations.push(id);
                    }
                }
            }
            for location in locations {
                source.absorb(&self.reaching(block, instruction, location));
            }
        }
        if op.effects().kind != SsaEffectKind::Pure {
            source.dynamic = true;
            self.remember(&mut source, Cause::Effect(block, instruction));
        }
        source
    }

    /// Derives contents and address requirements at the exact observation point.
    fn reaching(&mut self, block: usize, instruction: usize, id: usize) -> Source {
        if self.memory.location_by_id(id).is_none() {
            let mut source = Source::dynamic();
            self.remember(&mut source, Cause::Unknown(None));
            return source;
        }
        let version = self.memory.version_before_id(id, block, instruction);
        let immutable = self.memory.is_immutable_id(id);
        let mut source = Source::default();
        match version {
            Some(version) if immutable => {
                source.contents = true;
                self.remember(&mut source, Cause::Memory(id, version, MemoryOrigin::Image));
            }
            Some(version) => self.version(id, version, &mut source),
            None => {
                source.dynamic = true;
                self.remember(&mut source, Cause::Unknown(None));
            }
        }
        if let Some(location) = self.memory.location_by_id(id) {
            if source.contents {
                source.add_index_of(location);
            }
            if self.tracing {
                // Address identity remains required even when an exact store
                // makes scalar binding independent of the pointer's value.
                source.add_address_of(location, false);
            }
        }
        source
    }

    /// Returns the call `op` makes and the host's binding contract; `None`
    /// for anything that is not a call naming its method.
    ///
    /// What the host says of a method holds for every implementation of it, so
    /// a virtual or constrained call is described by the method it names.
    fn call<'op>(&self, op: &'op SsaOp<T>) -> Option<(CallBinding, &'op CallData<T>)> {
        let call = op.as_call()?;
        Some((self.oracle.call(call.callee.method()?), call))
    }

    /// Derives host-filled contents without changing ordinary argument indexing.
    fn filled(&self, block: usize, instr: usize, location: &MemoryLocation<T>) -> Option<Source> {
        let op = self
            .ssa
            .block(block)
            .and_then(|b| b.instructions().get(instr))
            .map(|i| i.op())?;
        let Some((CallBinding::Fills { argument }, call)) = self.call(op) else {
            return None;
        };
        let object = call.args.get(argument).copied()?;
        if Some(object) != location.base_object() {
            return None;
        }
        let mut source = Source::of(
            call.args
                .iter()
                .enumerate()
                .filter(|(position, _)| *position != argument)
                .map(|(_, value)| *value)
                .chain(call.implicit_inputs.iter().map(|input| input.value)),
        );
        source.contents = true;
        self.remember(&mut source, Cause::Call(block, instr));
        Some(source)
    }

    /// Walks canonical memory alternatives iteratively.
    ///
    /// Per-location version identities break phi cycles. A possible or partial
    /// write retains its previous contents; unknown arms retain other requirements.
    fn version(&mut self, id: usize, version: u32, source: &mut Source) {
        let mut pending = Vec::new();
        pending.push(version);
        let mut visited = FxHashSet::default();
        while let Some(version) = pending.pop() {
            if !self.tracing && source.dynamic {
                break;
            }
            if visited.contains(&version) {
                continue;
            }
            visited.insert(version);
            let Some(location) = self.memory.location_by_id(id) else {
                source.dynamic = true;
                self.remember(source, Cause::Unknown(None));
                continue;
            };
            match self.memory.defined_at(id, version) {
                Some(MemoryDefSite::Image) => {
                    source.contents = true;
                    self.remember(source, Cause::Memory(id, version, MemoryOrigin::Image));
                }
                Some(MemoryDefSite::Entry) => {
                    source.contents = true;
                    let contract = self.oracle.contents_at_entry(location);
                    source.add_time(contract.unwrap_or_else(|| {
                        BindingTime::of_contents_at_entry(location.region(self.context, None))
                    }));
                    self.remember(
                        source,
                        Cause::Memory(
                            id,
                            version,
                            if contract.is_some() {
                                MemoryOrigin::Host
                            } else {
                                MemoryOrigin::Entry
                            },
                        ),
                    );
                }
                Some(MemoryDefSite::Store { block, instr }) => {
                    self.remember(
                        source,
                        Cause::Memory(
                            id,
                            version,
                            MemoryOrigin::Write {
                                block,
                                instruction: instr,
                            },
                        ),
                    );
                    let stored = match self.memory.operation_at(block, instr) {
                        Some(MemoryOp::Store {
                            location: written,
                            value,
                            ..
                        }) if written.must_alias(location) => Some(*value),
                        _ => None,
                    };
                    if let Some(value) = stored {
                        source.values.push(Input::value(value));
                    } else if let Some(filled) = self.filled(block, instr, location) {
                        source.absorb(&filled);
                    } else {
                        source.dynamic = true;
                        self.remember(source, Cause::Effect(block, instr));
                        if let Some(MemoryOp::Store {
                            location: written,
                            value,
                            ..
                        }) = self.memory.operation_at(block, instr)
                        {
                            source.values.push(Input::value(*value));
                            source.add_address_of(written, true);
                        } else if let Some(op) = self
                            .ssa
                            .block(block)
                            .and_then(|b| b.instructions().get(instr))
                            .map(|i| i.op())
                        {
                            op.for_each_use(|value| source.values.push(Input::value(value)));
                        }
                        // The non-aliasing and preserved-byte alternatives,
                        // including read-modify-write inputs, still need this state.
                        if let Some(previous) = self.memory.version_before_id(id, block, instr) {
                            pending.push(previous);
                        } else {
                            self.remember(source, Cause::Unknown(None));
                        }
                    }
                    if self.tracing {
                        source.absorb(self.reach_control(block));
                    }
                }
                Some(MemoryDefSite::Phi { block }) => {
                    self.remember(
                        source,
                        Cause::Memory(id, version, MemoryOrigin::Merge { block }),
                    );
                    let mut operands = None;
                    for phi in self.memory.memory_phis(block) {
                        if phi.location == *location && phi.result_version == version {
                            let mut versions = Vec::new();
                            versions.extend(phi.operands.iter().map(|operand| operand.version));
                            operands = Some(versions);
                            break;
                        }
                    }
                    source.absorb(self.control(block));
                    if let Some(operands) = operands {
                        pending.extend(operands.into_iter().rev());
                    } else {
                        source.dynamic = true;
                        self.remember(source, Cause::Unknown(None));
                    }
                }
                Some(MemoryDefSite::ExceptionEntry { block, .. }) => {
                    self.remember(
                        source,
                        Cause::Memory(id, version, MemoryOrigin::Exception { block }),
                    );
                    source.dynamic = true;
                    self.remember(source, Cause::Exception(block));
                }
                None => {
                    source.dynamic = true;
                    self.remember(source, Cause::Unknown(None));
                }
            }
        }
    }

    /// Derives sparse control dependence without a quadratic block-bit matrix.
    ///
    /// Sparse pair walks stop on already visited suffixes of the canonical
    /// post-dominator tree.
    fn direct_controllers(&self) -> Vec<Vec<usize>> {
        let reverse = ReverseGraph::new(&self.cfg);
        let post = reverse.post_dominators();

        let mut rows = Vec::with_capacity(self.ssa.block_count());
        for _ in 0..self.ssa.block_count() {
            rows.push(Vec::new());
        }
        let mut seen = FxHashSet::default();
        for block in 0..self.ssa.block_count() {
            let successors = self.cfg.block_successors(block);
            let Some(first) = successors.first() else {
                continue;
            };
            let mut branches = false;
            for successor in successors {
                branches |= successor != first;
            }
            if !branches {
                continue;
            }
            let stop = post.immediate_post_dominator(NodeId::new(block));
            for successor in successors {
                let mut current = Some(NodeId::new(*successor));
                while let Some(node) = current.filter(|node| Some(*node) != stop) {
                    let pair = (node.index(), block);
                    if seen.contains(&pair) {
                        break;
                    }
                    seen.insert(pair);
                    if let Some(row) = rows.get_mut(node.index()) {
                        row.push(block);
                    }
                    current = post.immediate_post_dominator(node);
                }
            }
        }
        rows
    }

    /// Retains transitive guards deciding whether a block executes.
    ///
    /// Incoming merge selection is a separate question asked by phi sources.
    /// Reconverged guards do not constrain later independent constants.
    fn reach_control(&mut self, block: usize) -> &Source {
        if self.reach.contains_key(&block) {
            // Present: the entry is the kept one, and nothing is inserted.
            return self.reach.entry(block).or_default();
        }
        let mut source = Source::default();
        // Exceptional entry is a condition of its own. A protected return has
        // an exceptional CFG successor but no explicit normal-exit edge, so
        // ordinary post-dominance alone cannot represent that choice. Retain
        // handler/filter roots dominating this observation, including through
        // straight-line descendants; a normal/exceptional join drops them.
        for dominator in self.dominators.dominators(NodeId::new(block)) {
            if self.cfg.roots().is_flow_root(dominator.index()) {
                source.dynamic = true;
                self.remember(&mut source, Cause::Exception(dominator.index()));
            }
        }
        if self.controllers.is_none() {
            self.controllers = Some(self.direct_controllers());
        }
        let mut pending = Vec::new();
        pending.push(block);
        let mut seen = FxHashSet::default();
        while let Some(current) = pending.pop() {
            if seen.contains(&current) {
                continue;
            }
            seen.insert(current);
            let controllers = self
                .controllers
                .as_ref()
                .and_then(|rows| rows.get(current))
                .map_or(&[][..], Vec::as_slice);
            for controller in controllers {
                if !self.cfg.exceptional_successors(*controller).is_empty() {
                    source.dynamic = true;
                    self.remember(&mut source, Cause::Exception(*controller));
                }
                if let Some(terminator) = self
                    .ssa
                    .block(*controller)
                    .and_then(|decider| decider.control_terminator())
                {
                    terminator.for_each_use(|value| {
                        source.values.push(Input {
                            value,
                            role: DependencyRole::Control,
                            binding: false,
                        })
                    });
                } else {
                    source.dynamic = true;
                    self.remember(&mut source, Cause::Unknown(None));
                }
                pending.push(*controller);
            }
        }
        self.reach.entry(block).or_insert(source.distinct())
    }

    /// Retains the selectors choosing a merge's incoming predecessor state.
    ///
    /// The walk stops at the immediate dominator; instruction reachability is
    /// handled separately so selecting a phi arm does not lose enclosing guards.
    fn control(&mut self, block: usize) -> &Source {
        if self.control.contains_key(&block) {
            // Present: the entry is the kept one, and nothing is inserted.
            return self.control.entry(block).or_default();
        }
        let dominator = self
            .dominators
            .immediate_dominator(NodeId::new(block))
            .map(NodeId::index);
        let mut region = BTreeSet::new();
        let mut pending = Vec::new();
        pending.extend(self.cfg.block_predecessors(block).iter().copied());
        while let Some(current) = pending.pop() {
            if region.contains(&current) {
                continue;
            }
            region.insert(current);
            if Some(current) != dominator {
                pending.extend(self.cfg.block_predecessors(current).iter().copied());
            }
        }
        let mut source = Source::default();
        for current in region {
            let successors = self.cfg.block_successors(current);
            let Some(first) = successors.first() else {
                continue;
            };
            let mut branches = false;
            for successor in successors {
                branches |= successor != first;
            }
            if !branches {
                continue;
            }
            if !self.cfg.exceptional_successors(current).is_empty() {
                source.dynamic = true;
                self.remember(&mut source, Cause::Exception(current));
            }
            match self
                .ssa
                .block(current)
                .and_then(|decider| decider.control_terminator())
            {
                Some(terminator) => terminator.for_each_use(|value| {
                    source.values.push(Input {
                        value,
                        role: DependencyRole::Control,
                        binding: true,
                    })
                }),
                None => {
                    source.dynamic = true;
                    self.remember(&mut source, Cause::Unknown(None));
                }
            }
        }
        self.control.entry(block).or_insert(source.distinct())
    }
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashSet;

    use super::*;
    use crate::{
        ir::{
            function::{SsaDefSpec, SsaFunctionBuilder},
            variable::{DefSite, VariableOrigin},
        },
        testing::{MACHINE, MockTarget, MockType},
    };

    /// What `root` reaches, walked from scratch for this one request, with no
    /// closure kept: the oracle the kept closures are checked against.
    fn walked(analysis: &mut Dependencies<'_, MockTarget>, root: &Source) -> FxHashSet<Cause> {
        let mut causes: FxHashSet<Cause> = root
            .causes
            .iter()
            .map(|cause| match cause {
                Cause::Input(value, _) => Cause::Input(*value, DependencyRole::Value),
                other => *other,
            })
            .collect();
        let mut seen: FxHashSet<Node> = FxHashSet::default();
        let mut pending: Vec<Node> = root
            .values
            .iter()
            .map(|input| (input.value, input.role))
            .collect();
        while let Some((value, role)) = pending.pop() {
            if !seen.insert((value, role)) {
                continue;
            }
            let source = analysis.value_source(value);
            for cause in &source.causes {
                causes.insert(match cause {
                    Cause::Input(value, _) => Cause::Input(*value, role),
                    other => *other,
                });
            }
            for input in &source.values {
                let read = if input.role == DependencyRole::Value {
                    role
                } else {
                    input.role
                };
                pending.push((input.value, read));
            }
        }
        causes
    }

    /// **A table grows in few segments, and keeps every number.** Requests
    /// that each add one dependency leave a logarithmic number of segments,
    /// every dependency is read back by its number, and a table taken before
    /// a later request still reads what it held.
    #[test]
    fn a_table_grows_in_few_segments_and_keeps_every_number() {
        let unknown = |index: usize| Dependency::<MockTarget>::Unknown {
            value: Some(SsaVarId::from_index(index)),
        };
        let mut table = DependencyTable::<MockTarget>::default();
        let mut earlier = None;
        for index in 0..1000 {
            table.extend(vec![unknown(index)]);
            if index == 499 {
                earlier = Some(table.clone());
            }
            assert!(
                table.segments.len() <= (index + 1).ilog2() as usize + 1,
                "{} segments for {} dependencies",
                table.segments.len(),
                index + 1
            );
        }
        table.extend(Vec::new());
        assert_eq!(table.len(), 1000);
        for index in 0..1000 {
            assert!(matches!(
                table.get(index),
                Some(Dependency::Unknown { value: Some(value) }) if value.index() == index
            ));
        }
        assert!(table.get(1000).is_none());
        let earlier = earlier.expect("the table at 500");
        assert_eq!(earlier.len(), 500);
        for index in 0..500 {
            assert!(matches!(
                earlier.get(index),
                Some(Dependency::Unknown { value: Some(value) }) if value.index() == index
            ));
        }
        assert!(earlier.get(500).is_none());
    }

    /// **A chain as long as a binary makes it is walked on a small stack.** A
    /// hundred thousand values each read the one before; the last one's
    /// closure is found on a 256 KiB thread and names the chain's one input.
    #[test]
    fn a_long_chain_is_walked_on_a_small_stack() {
        let reached = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
                let key = builder.function_mut().create_variable(
                    VariableOrigin::Argument(0),
                    0,
                    DefSite::entry(),
                    MockType::I32,
                );
                let last = builder
                    .in_block(0, |block| {
                        let mut last = key;
                        for _ in 0..100_000 {
                            last = block.add(SsaDefSpec::tmp(MockType::I32), last, key)?;
                        }
                        block.ret(Some(last))?;
                        Ok(last)
                    })
                    .expect("the chain");
                let ssa = builder.finish().expect("the function");
                let context = AnalysisContext::new(MACHINE);
                let mut analysis = Dependencies::analyze(&ssa, &context, &());
                let signature = analysis
                    .request(Observation::Value(last, DependencyRole::Value))
                    .expect("a signature");
                let inputs: Vec<_> = signature
                    .dependencies()
                    .map(|dependency| match dependency {
                        Dependency::Input { value, role } => Some((*value, *role)),
                        _ => None,
                    })
                    .collect();
                (inputs, key)
            })
            .expect("a thread")
            .join()
            .expect("no overflow");
        let (inputs, key) = reached;
        assert_eq!(inputs, vec![Some((key, DependencyRole::Value))]);
    }

    /// **A load joins what its stores are reached under once.** A load
    /// reached by sixteen stores, each behind its own selector and all of them
    /// behind the same eight guards, holds each guard once among its edges,
    /// not once per store: a source is a set, and every walk over it reads
    /// what it holds.
    #[test]
    fn a_load_holds_the_guards_its_stores_share_once() {
        const GUARDS: usize = 8;
        const STORES: usize = 16;
        let arguments = 1 + GUARDS + 2 * STORES;
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(arguments, 0);
        let mut next_argument = 0u16;
        let mut parameter = |ty: MockType| {
            let variable = builder.function_mut().create_variable(
                VariableOrigin::Argument(next_argument),
                0,
                DefSite::entry(),
                ty,
            );
            next_argument += 1;
            variable
        };
        let pointer = parameter(MockType::Ptr);
        let guards: Vec<SsaVarId> = (0..GUARDS).map(|_| parameter(MockType::I32)).collect();
        let selectors: Vec<SsaVarId> = (0..STORES).map(|_| parameter(MockType::I32)).collect();
        let stored: Vec<SsaVarId> = (0..STORES).map(|_| parameter(MockType::I32)).collect();
        // Guards 0..GUARDS, then each store's selector and store block, then
        // the load and the way out.
        let selector_block = |store: usize| GUARDS + 2 * store;
        let load_block = GUARDS + 2 * STORES;
        let exit = load_block + 1;
        builder.ensure_block(exit);
        for (index, guard) in guards.iter().enumerate() {
            builder
                .in_block(index, |block| block.branch(*guard, index + 1, exit))
                .expect("a guard");
        }
        for store in 0..STORES {
            let after = selector_block(store + 1);
            builder
                .in_block(selector_block(store), |block| {
                    block.branch(selectors[store], selector_block(store) + 1, after)
                })
                .expect("a selector");
            builder
                .in_block(selector_block(store) + 1, |block| {
                    block.store_indirect(pointer, stored[store], MockType::I32)?;
                    block.jump(after)
                })
                .expect("a store");
        }
        let loaded = builder
            .in_block(load_block, |block| {
                let loaded =
                    block.load_indirect(SsaDefSpec::tmp(MockType::I32), pointer, MockType::I32)?;
                block.ret(Some(loaded))?;
                Ok(loaded)
            })
            .expect("the load");
        builder
            .in_block(exit, |block| block.ret(None))
            .expect("the way out");
        let ssa = builder.finish().expect("the function");
        let context = AnalysisContext::new(MACHINE);
        let mut analysis = Dependencies::analyze(&ssa, &context, &());

        let source = analysis.value_source(loaded);
        let held: FxHashSet<Input> = source.values.iter().copied().collect();
        assert_eq!(
            source.values.len(),
            held.len(),
            "each edge once: {source:?}"
        );
        for guard in &guards {
            let reads = source
                .values
                .iter()
                .filter(|input| input.value == *guard && input.role == DependencyRole::Control)
                .count();
            assert_eq!(reads, 1, "guard {guard:?} is read once");
        }
        for value in &stored {
            assert!(
                source.values.iter().any(|input| input.value == *value),
                "every store's value reaches the load"
            );
        }
        let causes: FxHashSet<Cause> = source.causes.iter().copied().collect();
        assert_eq!(source.causes.len(), causes.len(), "each identity once");
    }

    /// **A kept closure is what a walk of its own finds.** Over a loop that
    /// carries a counter round, a load through a parameter and a branch on
    /// both, every value read in every role names exactly the dependencies a
    /// walk from scratch reaches, whichever values were asked about before.
    #[test]
    fn a_kept_closure_is_what_a_walk_of_its_own_finds() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(3, 0);
        let mut parameter = |index: u16, ty: MockType| {
            builder.function_mut().create_variable(
                VariableOrigin::Argument(index),
                0,
                DefSite::entry(),
                ty,
            )
        };
        let (bound, key, pointer) = (
            parameter(0, MockType::I32),
            parameter(1, MockType::I32),
            parameter(2, MockType::Ptr),
        );
        builder.ensure_block(2);
        let tmp = || SsaDefSpec::tmp(MockType::I32);
        let start = builder
            .in_block(0, |block| {
                let start = block.load_indirect(tmp(), pointer, MockType::I32)?;
                block.jump(1)?;
                Ok(start)
            })
            .expect("the entry");
        let (counter, next) = builder
            .in_block(1, |block| {
                let counter = block.empty_phi(tmp())?;
                let next = block.add(tmp(), counter, key)?;
                let more = block.clt(tmp(), next, bound, false)?;
                block.branch(more, 1, 2)?;
                Ok((counter, next))
            })
            .expect("the loop");
        builder
            .add_phi_operand(1, counter, 0, start)
            .expect("the way in");
        builder
            .add_phi_operand(1, counter, 1, next)
            .expect("the way round");
        builder
            .in_block(2, |block| block.ret(Some(next)))
            .expect("the way out");
        let ssa = builder.finish().expect("the function");
        let context = AnalysisContext::new(MACHINE);
        let mut analysis = Dependencies::analyze(&ssa, &context, &());
        let roles = [
            DependencyRole::Value,
            DependencyRole::Address,
            DependencyRole::Control,
            DependencyRole::Bound,
        ];
        for value in [next, counter, start, key, bound, pointer] {
            for role in roles {
                let root = analysis
                    .observation_source(Observation::Value(value, role))
                    .expect("a value of the function");
                let expected = walked(&mut analysis, &root);
                let signature = analysis.trace(root).expect("a signature");
                let named: FxHashSet<usize> = signature.members.iter().collect();
                let walked: FxHashSet<usize> = expected
                    .iter()
                    .map(|cause| *analysis.numbers.get(cause).expect("every cause numbered"))
                    .collect();
                assert_eq!(named, walked, "{value:?} read as {role:?}");
                assert_eq!(signature.len(), walked.len());
            }
        }
    }
}
