// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Physical optimizer interfaces.

use std::fmt::Debug;
use std::sync::Arc;

use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::operator_statistics::StatisticsRegistry;

/// Context available to physical optimizer rules.
///
/// This trait provides access to configuration options and an optional statistics
/// registry for enhanced statistics lookup.
pub trait PhysicalOptimizerContext: Send + Sync {
    /// Returns the configuration options.
    fn config_options(&self) -> &ConfigOptions;

    /// Returns the statistics registry for enhanced statistics lookup.
    ///
    /// Returns `None` if no registry is configured, in which case rules
    /// should fall back to using [`ExecutionPlan::partition_statistics`].
    fn statistics_registry(&self) -> Option<&StatisticsRegistry> {
        None
    }
}

/// `PhysicalOptimizerRule` transforms one [`ExecutionPlan`] into another which
/// computes the same results, but in a potentially more efficient way.
///
/// Use [`SessionState::add_physical_optimizer_rule`] to register additional
/// `PhysicalOptimizerRule`s.
///
/// [`SessionState::add_physical_optimizer_rule`]: https://docs.rs/datafusion/latest/datafusion/execution/session_state/struct.SessionState.html#method.add_physical_optimizer_rule
pub trait PhysicalOptimizerRule: Debug + std::any::Any {
    /// Rewrite `plan` to an optimized form.
    ///
    /// This is the primary optimization method. For rules that need access to
    /// the statistics registry, override [`optimize_with_context`](Self::optimize_with_context) instead.
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>>;

    /// Rewrite `plan` with access to extended context (statistics registry, etc.).
    ///
    /// Override this method if you need access to the statistics registry for
    /// enhanced statistics lookup. The default implementation simply calls
    /// [`optimize`](Self::optimize) with the config options from the context.
    fn optimize_with_context(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        context: &dyn PhysicalOptimizerContext,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.optimize(plan, context.config_options())
    }

    /// A human readable name for this optimizer rule
    fn name(&self) -> &str;

    /// A flag to indicate whether the physical planner should validate that the rule will not
    /// change the schema of the plan after the rewriting.
    /// Some of the optimization rules might change the nullable properties of the schema
    /// and should disable the schema check.
    fn schema_check(&self) -> bool;

    /// Whether this rule's output is a pure function of the plan and the
    /// configuration: no clocks, no randomness, no external mutable state.
    ///
    /// Within one optimization run, once a deterministic rule has been
    /// *observed* to return a plan unchanged, a later call handing it that
    /// same plan is skipped outright, since it would provably do the same
    /// nothing. Nothing is ever re-applied speculatively, so this is safe
    /// for rules that are not idempotent: a rule that oscillates between two
    /// forms is deterministic, never records a fixpoint, and is simply never
    /// skipped. Wrappers must delegate this, or the skip silently dies.
    fn deterministic(&self) -> bool {
        false
    }
}
