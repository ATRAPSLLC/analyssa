//! Creating variables.

use crate::{
    facts::Derivation,
    ir::{
        function::editor::SsaEditor,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    target::Target,
};

impl<'a, T: Target> SsaEditor<'a, T> {
    /// Creates a variable with an explicit type.
    ///
    /// The editor-facing equivalent of [`SsaFunction::create_variable`](crate::ir::function::SsaFunction::create_variable), for a
    /// caller that knows the type the new value carries rather than wanting
    /// the origin registry's answer - a phi minted to stand for a value the
    /// edges already carry takes their type, not the origin's. Creating a
    /// variable does not by itself require repair, because the caller must
    /// still insert the corresponding definition before the edit boundary.
    ///
    /// # Arguments
    ///
    /// * `origin` - Source-level origin the new variable descends from.
    /// * `version` - SSA version number for that origin.
    /// * `def_site` - Location where the caller will place the definition.
    /// * `var_type` - The type the new value carries.
    ///
    /// # Returns
    ///
    /// The id of the newly registered variable.
    #[must_use]
    pub fn create_variable(
        &mut self,
        origin: VariableOrigin,
        version: u32,
        def_site: DefSite,
        var_type: T::Type,
    ) -> SsaVarId {
        self.ssa
            .create_variable(origin, version, def_site, var_type)
    }

    /// Creates a variable for a value that is no existing value under a new
    /// id: the result of a computation a pass introduces (`base + offset`), or
    /// a constant it materialises.
    ///
    /// Such a value has no parent, so no host fact follows it - what a host
    /// knows about `base` is not true of `base + offset`. A value that *does*
    /// stand for an existing one is created through
    /// [`Self::derive_variable_for_origin`] (or, on the function,
    /// [`SsaFunction::create_variable_derived`](crate::ir::function::SsaFunction::create_variable_derived) and
    /// [`SsaFunction::create_variable_joined`](crate::ir::function::SsaFunction::create_variable_joined)), which carry the host's facts
    /// across; naming the fresh case separately is
    /// what lets a pass's choice between the two be read at the call site.
    ///
    /// # Arguments
    ///
    /// * `origin` - Source-level origin the new variable descends from.
    /// * `version` - SSA version number for that origin.
    /// * `def_site` - Location where the caller will place the definition.
    /// * `var_type` - The type the new value carries.
    ///
    /// # Returns
    ///
    /// The id of the newly registered variable.
    #[must_use]
    pub fn create_fresh_variable(
        &mut self,
        origin: VariableOrigin,
        version: u32,
        def_site: DefSite,
        var_type: T::Type,
    ) -> SsaVarId {
        self.ssa
            .create_variable(origin, version, def_site, var_type)
    }

    /// Creates a variable that stands for `parent`, typed from the origin type
    /// registry.
    ///
    /// The editor-facing equivalent of
    /// [`SsaFunction::derive_variable_for_origin`](crate::ir::function::SsaFunction::derive_variable_for_origin): the host's facts come from
    /// `parent`. Like [`Self::create_variable`], creating it does not
    /// by itself require repair.
    ///
    /// # Arguments
    ///
    /// * `parent` - The value the new one stands for.
    /// * `origin` - Source-level origin the new variable descends from.
    /// * `version` - SSA version number for that origin.
    /// * `def_site` - Location where the caller will place the definition.
    /// * `cause` - What derived it.
    ///
    /// # Returns
    ///
    /// The new variable's id.
    pub fn derive_variable_for_origin(
        &mut self,
        parent: SsaVarId,
        origin: VariableOrigin,
        version: u32,
        def_site: DefSite,
        cause: Derivation,
    ) -> SsaVarId {
        self.ssa
            .derive_variable_for_origin(parent, origin, version, def_site, cause)
    }
}
