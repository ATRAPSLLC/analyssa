//! Requested byte and scalar facts after an admitted transfer and parallel phi assignment.

use std::{
    collections::{BTreeMap, BTreeSet},
    ptr, result,
};

use crate::{
    Error, Result,
    analysis::{
        cfg::SsaCfg,
        context::AnalysisContext,
        execution::{
            ObservationSizeRefusal, access::Access, edgescalar::ScalarLayout, facts::WrittenRanges,
            state::State, value::Address,
        },
    },
    graph::{NodeId, RootedGraph, algorithms::compute_dominators},
    ir::{
        function::SsaFunction, instruction::InstrId, ops::SsaOp, value::ConstValue,
        variable::SsaVarId,
    },
    pointer::PointerSize,
    target::Target,
};

/// One normal target edge in an immutable producer body.
///
/// Equal-target branch and switch arms share this identity. It does not identify
/// a predicate arm or establish that an explored alternative is concretely feasible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct EdgeSite {
    /// Source predecessor block.
    pub from: usize,
    /// Source terminator position within that block.
    pub instruction: usize,
    /// Stable source terminator identity, when the source was stamped.
    pub instruction_id: Option<InstrId>,
    /// Target block whose phis have just been assigned.
    pub to: usize,
}

impl EdgeSite {
    /// Identifies an existing direct normal transfer in this source snapshot.
    ///
    /// # Errors
    /// Refuses missing blocks and unsupported or unrelated transfers.
    pub fn of<T: Target>(body: &SsaFunction<T>, from: usize, to: usize) -> Result<Self> {
        let block = body
            .block(from)
            .ok_or_else(|| Error::new("missing edge source"))?;
        let instruction = block
            .instructions()
            .len()
            .checked_sub(1)
            .ok_or_else(|| Error::new("edge source has no terminator"))?;
        let terminator = block
            .instructions()
            .get(instruction)
            .ok_or_else(|| Error::new("missing edge terminator"))?;
        let transfers = match terminator.op() {
            SsaOp::Jump { target } => *target == to,
            SsaOp::Branch {
                true_target,
                false_target,
                ..
            }
            | SsaOp::BranchCmp {
                true_target,
                false_target,
                ..
            }
            | SsaOp::BranchFlags {
                true_target,
                false_target,
                ..
            } => *true_target == to || *false_target == to,
            SsaOp::Switch {
                targets, default, ..
            } => *default == to || targets.contains(&to),
            _ => false,
        };
        if !transfers || body.block(to).is_none() {
            return Err(Error::new("requested edge is not a direct normal transfer"));
        }
        Ok(Self {
            from,
            instruction,
            instruction_id: terminator.id(),
            to,
        })
    }
}

/// One bounded prefix requested through a pointer visible at target entry.
pub struct EdgeByteRequest<'a, T: Target> {
    /// Exact immutable body which execution must enter before this can be observed.
    pub body: &'a SsaFunction<T>,
    /// Target edge in that body.
    pub site: EdgeSite,
    /// Pointer value, including a target phi assigned by the transfer.
    pub pointer: SsaVarId,
    /// Maximum prefix extent; this is not an object-size or lifetime assertion.
    pub maximum_bytes: usize,
}

/// One fixed scalar requested after the target's parallel phi assignment.
pub struct EdgeScalarRequest<'a, T: Target> {
    /// Exact immutable body which execution must actually enter.
    pub body: &'a SsaFunction<T>,
    /// Admitted target edge in that body.
    pub site: EdgeSite,
    /// Source SSA identity available at this target entry.
    pub value: SsaVarId,
}

struct PlanRequest<'a, T: Target> {
    body: &'a SsaFunction<T>,
    site: EdgeSite,
    value: SsaVarId,
    maximum_bytes: Option<usize>,
}

/// Owner admission bounds checked before request-plan allocation or type queries.
#[derive(Debug, Clone, Copy)]
pub struct EdgeRequestLimits {
    /// Maximum distinct body/edge sites.
    pub sites: usize,
    /// Maximum submitted operands, including duplicates before canonicalization.
    pub requests: usize,
}

struct SiteRequests {
    site: EdgeSite,
    prefixes: BTreeMap<SsaVarId, usize>,
    scalars: BTreeMap<SsaVarId, ScalarLayout>,
}

struct BodyRequests<'a, T: Target> {
    body: &'a SsaFunction<T>,
    sites: BTreeMap<(usize, usize), SiteRequests>,
}

/// A bounded, validated observation plan tied to borrowed immutable bodies.
///
/// Construct once in the owning analysis. This plan does not schedule execution,
/// create invocation contexts or prove that any requested edge is reached.
/// Body addresses index only this immutable borrowed execution lifetime and are
/// checked against the retained body reference. They are never persisted source
/// identities or cache keys. Integer address indices avoid imposing raw-pointer
/// Send/Sync restrictions beyond those of the borrowed source bodies themselves.
pub struct EdgeObservationPlan<'a, T: Target> {
    ptr_size: PointerSize,
    bodies: BTreeMap<usize, BodyRequests<'a, T>>,
}

impl<'a, T: Target> EdgeObservationPlan<'a, T> {
    /// Validates normal source edges, pointer types and incoming-edge visibility.
    ///
    /// Duplicate body/edge/pointer requests use their maximum extent. The submitted
    /// operand bound is checked before allocating, so duplicates cannot hide an
    /// oversized input. Source bodies must already satisfy ordinary SSA invariants.
    ///
    /// # Errors
    /// Refuses admission bounds, stale source sites, unsupported pointer types,
    /// absent definitions and pointers unavailable on the edge.
    pub fn new(
        requests: &[EdgeByteRequest<'a, T>],
        ptr_size: PointerSize,
        limits: EdgeRequestLimits,
    ) -> Result<Self> {
        Self::with_scalars(requests, &[], ptr_size, limits)
    }

    /// Admits byte and fixed scalar requests under one combined metadata bound.
    ///
    /// Scalars are point-qualified observations, never definition replacement facts.
    ///
    /// # Errors
    /// Refuses oversized combined requests, invalid source edges, unavailable values,
    /// unsupported scalar layouts and the byte request failures of [`Self::new`].
    pub fn with_scalars(
        bytes: &[EdgeByteRequest<'a, T>],
        scalars: &[EdgeScalarRequest<'a, T>],
        ptr_size: PointerSize,
        limits: EdgeRequestLimits,
    ) -> Result<Self> {
        if bytes
            .len()
            .checked_add(scalars.len())
            .is_none_or(|count| count > limits.requests)
        {
            return Err(Error::new("edge observation operand admission bound"));
        }
        let requests = || {
            bytes
                .iter()
                .map(|request| PlanRequest {
                    body: request.body,
                    site: request.site,
                    value: request.pointer,
                    maximum_bytes: Some(request.maximum_bytes),
                })
                .chain(scalars.iter().map(|request| PlanRequest {
                    body: request.body,
                    site: request.site,
                    value: request.value,
                    maximum_bytes: None,
                }))
        };
        // The raw operand count bounds this temporary set before allocation.
        // Admit each distinct site before inserting it; no type graphs are cloned.
        let mut sites = BTreeSet::new();
        for request in requests() {
            let key = (ptr::from_ref(request.body).addr(), request.site);
            if !sites.contains(&key) {
                if sites.len() >= limits.sites {
                    return Err(Error::new("edge observation site admission bound"));
                }
                sites.insert(key);
            }
        }
        let mut plan = Self {
            ptr_size,
            bodies: BTreeMap::new(),
        };
        let mut dominators = BTreeMap::new();
        let mut layouts = BTreeMap::new();
        for request in requests() {
            let body = request.body;
            let body_key = ptr::from_ref(body).addr();
            let edge_key = (request.site.from, request.site.to);
            if !plan
                .bodies
                .get(&body_key)
                .is_some_and(|body| body.sites.contains_key(&edge_key))
                && EdgeSite::of(body, request.site.from, request.site.to)? != request.site
            {
                return Err(Error::new("unsupported or stale requested edge"));
            }
            let variable = body
                .variable(request.value)
                .ok_or_else(|| Error::new("missing edge value"))?;
            let layout = match layouts.get(&(body_key, request.value)) {
                Some(layout) => *layout,
                None => {
                    let layout = T::type_layout(variable.var_type(), ptr_size)
                        .ok_or_else(|| Error::new("unsupported edge value type"))?;
                    layouts.insert((body_key, request.value), layout);
                    layout
                }
            };
            let scalar = if request.maximum_bytes.is_some() {
                if layout.is_runtime_sized()
                    || layout.is_boolean()
                    || layout.vector_descriptor().is_some()
                    || layout.vector_mask_descriptor().is_some()
                    || !(layout.is_pointer() || layout.is_integer())
                    || layout.storage_bytes(ptr_size) != Some(ptr_size.bytes())
                {
                    return Err(Error::new("edge pointer is not a pointer-width scalar"));
                }
                None
            } else {
                Some(
                    ScalarLayout::of(layout)
                        .ok_or_else(|| Error::new("unsupported edge scalar layout"))?,
                )
            };
            let definition = variable.def_site();
            let target_phi = body.block(request.site.to).is_some_and(|block| {
                block
                    .phi_nodes()
                    .iter()
                    .any(|phi| phi.result() == request.value)
            });
            let entry = variable.origin().is_caller_supplied()
                && definition.block() == 0
                && definition.index().is_none();
            let defined = entry
                || match definition.index() {
                    Some(_) => body.recorded_definition(request.value).is_some(),
                    None => body.block(definition.block()).is_some_and(|block| {
                        block
                            .phi_nodes()
                            .iter()
                            .any(|phi| phi.result() == request.value)
                    }),
                };
            if !defined {
                return Err(Error::new("edge value has no recorded SSA definition"));
            }
            if !target_phi {
                let key = ptr::from_ref(body).addr();
                let tree = dominators.entry(key).or_insert_with(|| {
                    let cfg = SsaCfg::from_ssa(body);
                    compute_dominators(&cfg, cfg.entry())
                });
                if !tree.dominates(
                    NodeId::new(definition.block()),
                    NodeId::new(request.site.from),
                ) || definition.block() == request.site.from
                    && definition
                        .index()
                        .is_some_and(|index| index >= request.site.instruction)
                {
                    return Err(Error::new("edge value does not reach target entry"));
                }
            }
            let records = plan.bodies.entry(body_key).or_insert_with(|| BodyRequests {
                body,
                sites: BTreeMap::new(),
            });
            let record = records
                .sites
                .entry(edge_key)
                .or_insert_with(|| SiteRequests {
                    site: request.site,
                    prefixes: BTreeMap::new(),
                    scalars: BTreeMap::new(),
                });
            if record.site != request.site {
                return Err(Error::new("incompatible duplicate edge identity"));
            }
            if let Some(maximum) = request.maximum_bytes {
                record
                    .prefixes
                    .entry(request.value)
                    .and_modify(|extent| *extent = (*extent).max(maximum))
                    .or_insert(maximum);
            } else if let Some(layout) = scalar {
                record.scalars.entry(request.value).or_insert(layout);
            }
        }
        Ok(plan)
    }

    /// Returns the pointer width every request in this plan was admitted for.
    pub const fn pointer_size(&self) -> PointerSize {
        self.ptr_size
    }

    fn requests(&self, body: &SsaFunction<T>, from: usize, to: usize) -> Option<&SiteRequests> {
        let requests = self.bodies.get(&ptr::from_ref(body).addr())?;
        if !ptr::eq(requests.body, body) {
            return None;
        }
        let record = requests.sites.get(&(from, to))?;
        Some(record)
    }
}

/// A visited pointer prefix, met across all visits to this target edge and scope.
#[derive(Debug, Clone)]
pub struct EdgeBytes {
    /// Requested source-domain pointer identity.
    pub pointer: SsaVarId,
    /// Canonical requested maximum, distinct from retained prefix length.
    pub maximum_bytes: usize,
    /// Common address, absent after an unknown or conflicting visit.
    pub address: Option<Address>,
    /// Common prefix, possibly empty; an empty visited record is not never-visited.
    pub bytes: Vec<u8>,
    /// Which of the bytes the run wrote, on every visit; the rest are the
    /// image's own, unchanged.
    pub written: WrittenRanges,
}

/// Target-entry bytes and fixed scalars after parallel phis and before arrival joining.
#[derive(Debug, Clone)]
pub struct EdgeObservation<T: Target> {
    /// Actual admitted normal target edge in this producer snapshot.
    pub site: EdgeSite,
    /// Requested pointers in stable SSA-variable order, including unavailable visits.
    pub prefixes: Vec<EdgeBytes>,
    /// Requested fixed scalars, including visited values whose exact fact was lost.
    pub scalars: Vec<EdgeScalar<T>>,
}

/// One scalar at a visited edge, separate from all-definition replacement facts:
/// what holds as control passes along the edge, what the conditions of the
/// branches that led there establish of it included.
#[derive(Debug, Clone)]
pub struct EdgeScalar<T: Target> {
    /// Source SSA value identity.
    pub value: SsaVarId,
    /// Exact typed value common to all visits, or unavailable after any disagreement.
    pub constant: Option<ConstValue<T>>,
}

/// One trial's owner-supplied edge-payload reservation; execution continues on overflow.
#[derive(Default)]
pub struct EdgeAllowance {
    remaining: usize,
    pub exhausted: bool,
}

impl EdgeAllowance {
    pub const fn new(maximum: usize) -> Self {
        Self {
            remaining: maximum,
            exhausted: false,
        }
    }
}

/// Per-domain records with unavailable visits retained monotonically.
#[derive(Debug, Clone)]
pub struct EdgeRecords<T: Target> {
    sites: BTreeMap<EdgeSite, EdgeObservation<T>>,
}

impl<T: Target> Default for EdgeRecords<T> {
    fn default() -> Self {
        Self {
            sites: BTreeMap::new(),
        }
    }
}

impl<T: Target> EdgeRecords<T> {
    /// Meets the admitted transfer against its actual state after parallel phis.
    pub fn observe(
        &mut self,
        plan: &EdgeObservationPlan<'_, T>,
        body: &SsaFunction<T>,
        edge: (usize, usize),
        state: &State<T>,
        context: &AnalysisContext<'_>,
        allowance: &mut EdgeAllowance,
    ) {
        let Some(frame) = state.frame() else {
            return;
        };
        let memory = state.memory();
        let access = Access::new(context);
        let machine = context.machine();
        let Some(requests) = plan.requests(body, edge.0, edge.1) else {
            return;
        };
        let site = requests.site;
        if let Some(known) = self.sites.get_mut(&site) {
            for scalar in &mut known.scalars {
                let Some(held) = scalar.constant.as_ref() else {
                    continue;
                };
                let exact = requests.scalars.get(&scalar.value).and_then(|layout| {
                    frame
                        .held_value(scalar.value)
                        .and_then(|value| layout.read(value, machine))
                });
                if exact.as_ref() != Some(held) {
                    allowance.remaining = allowance.remaining.saturating_add(held.payload_bytes());
                    scalar.constant = None;
                }
            }
            for prefix in &mut known.prefixes {
                let previous = prefix.bytes.len();
                let address = frame
                    .held_value(prefix.pointer)
                    .and_then(|value| access.address_of(value, None));
                if address != prefix.address {
                    prefix.address = None;
                    prefix.bytes.clear();
                    prefix.written = WrittenRanges::default();
                } else if let Some(address) = address {
                    access.intersect_known_run(memory, address, &mut prefix.bytes);
                    let written = memory.written(address, prefix.bytes.len());
                    prefix.written = prefix.written.meet(&written, 0);
                    prefix.written.truncate(prefix.bytes.len());
                }
                allowance.remaining = allowance
                    .remaining
                    .saturating_add(previous.saturating_sub(prefix.bytes.len()));
            }
            return;
        }
        let mut prefixes = Vec::with_capacity(requests.prefixes.len());
        for (&pointer, &maximum_bytes) in &requests.prefixes {
            let address = frame
                .held_value(pointer)
                .and_then(|value| access.address_of(value, None));
            let bytes = if !allowance.exhausted
                && let Some(address) = address
            {
                let limit =
                    address.linear_prefix_len(maximum_bytes.min(access.observation_limit()));
                match access.known_run_within(memory, address, limit, allowance.remaining) {
                    Some(bytes) => {
                        allowance.remaining = allowance.remaining.saturating_sub(bytes.len());
                        bytes
                    }
                    None => {
                        allowance.exhausted = true;
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            };
            let written = address.map_or_else(WrittenRanges::default, |address| {
                memory.written(address, bytes.len())
            });
            prefixes.push(EdgeBytes {
                pointer,
                maximum_bytes,
                address,
                bytes,
                written,
            });
        }
        let mut scalars = Vec::with_capacity(requests.scalars.len());
        for (&value, &layout) in &requests.scalars {
            let mut constant = (!allowance.exhausted)
                .then(|| {
                    frame
                        .held_value(value)
                        .and_then(|held| layout.read(held, machine))
                })
                .flatten();
            if let Some(exact) = &constant {
                let bytes = exact.payload_bytes();
                if bytes > allowance.remaining {
                    allowance.exhausted = true;
                    constant = None;
                } else {
                    allowance.remaining = allowance.remaining.saturating_sub(bytes);
                }
            }
            scalars.push(EdgeScalar { value, constant });
        }
        self.sites.insert(
            site,
            EdgeObservation {
                site,
                prefixes,
                scalars,
            },
        );
    }

    /// Counts retained payload by borrowing every requested point, including unknown markers.
    pub fn payload_bytes_within(
        &self,
        maximum_bytes: usize,
        cancelled: &dyn Fn() -> bool,
    ) -> result::Result<usize, ObservationSizeRefusal> {
        let mut bytes = 0usize;
        for record in self.sites.values() {
            if cancelled() {
                return Err(ObservationSizeRefusal::Cancelled);
            }
            for prefix in &record.prefixes {
                if cancelled() {
                    return Err(ObservationSizeRefusal::Cancelled);
                }
                bytes = bytes
                    .checked_add(prefix.bytes.len())
                    .filter(|total| *total <= maximum_bytes)
                    .ok_or(ObservationSizeRefusal::OutputLimit)?;
            }
            for scalar in &record.scalars {
                if cancelled() {
                    return Err(ObservationSizeRefusal::Cancelled);
                }
                if let Some(constant) = &scalar.constant {
                    bytes = bytes
                        .checked_add(constant.payload_bytes())
                        .filter(|total| *total <= maximum_bytes)
                        .ok_or(ObservationSizeRefusal::OutputLimit)?;
                }
            }
        }
        Ok(bytes)
    }

    pub fn into_observations(self) -> Vec<EdgeObservation<T>> {
        self.sites.into_values().collect()
    }
}

#[cfg(test)]
#[path = "edges_tests.rs"]
mod tests;
