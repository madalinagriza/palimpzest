"""
PrivacyAwareExecutionStrategy — drop-in replacement for PZ's sequential
execution strategy that routes each LLM operator through execute_with_routing()
before invocation.

Usage
-----
    from privacy.privacy_execution_strategy import create_privacy_processor
    import palimpzest as pz

    config = pz.QueryProcessorConfig(
        policy=pz.MaxQuality(),
        execution_strategy="sequential",
        sentinel_execution_strategy=None,
        available_models=[pz.Model.GPT_4o],
        progress=False,
        verbose=False,
        k=1, j=1, sample_budget=5,
    )

    processor = create_privacy_processor(plan, config)
    result = processor.execute()

    # Print per-run routing summary for the benchmark table
    print(processor.execution_strategy.router.stats.summary())

No PZ source files are modified.
"""
from __future__ import annotations

import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from palimpzest.core.elements.records import DataRecord
from palimpzest.core.models import PlanStats
from palimpzest.query.execution.single_threaded_execution_strategy import (
    SequentialSingleThreadExecutionStrategy,
)
from palimpzest.query.operators.aggregate import AggregateOp
from palimpzest.query.operators.join import JoinOp
from palimpzest.query.operators.limit import LimitScanOp
from palimpzest.query.operators.scan import ContextScanOp, ScanPhysicalOp
from palimpzest.query.optimizer.plan import PhysicalPlan
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.query.processor.query_processor_factory import QueryProcessorFactory

from routing_stub import PrivacyRouter, RoutingGranularity, RouteDecision, execute_with_routing

logger = logging.getLogger(__name__)


class PrivacyAwareExecutionStrategy(SequentialSingleThreadExecutionStrategy):
    """
    Sequential execution strategy with a privacy routing hook.

    Supports three routing granularities (RoutingGranularity):
      OPERATOR  — scan only get_input_fields() per operator call (default)
      FIELD     — scan all input_schema fields per operator call
      DOCUMENT  — scan all fields once per record, reuse for all operators

    For every operator that makes LLM calls (get_model_name() is not None),
    execute_with_routing() detects PII, optionally swaps the model, and logs.
    Non-LLM operators (scans, limits) are passed through unchanged.
    """

    def __init__(
        self,
        *args,
        router: PrivacyRouter | None = None,
        granularity: RoutingGranularity = RoutingGranularity.OPERATOR,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.router = router or PrivacyRouter()
        self.granularity = granularity
        # Cache for document-level routing: record_key → RouteDecision
        self._doc_cache: dict[str, RouteDecision] = {}

    @staticmethod
    def _record_key(input_record) -> str:
        """Stable key for a record: uses record_id if present, else object id."""
        return str(getattr(input_record, "record_id", None) or id(input_record))

    def _invoke_operator(self, operator, input_record):
        """
        Route and invoke a single operator according to self.granularity.
        Non-LLM operators are always passed through directly.
        """
        if operator.get_model_name() is None:
            return operator(input_record)

        if self.granularity == RoutingGranularity.OPERATOR:
            # Default: scan only the fields this operator reads (respects depends_on)
            return execute_with_routing(operator, input_record, self.router)

        elif self.granularity == RoutingGranularity.FIELD:
            # Scan all fields in the input schema regardless of depends_on
            all_fields = (
                list(operator.input_schema.model_fields)
                if operator.input_schema is not None
                else operator.get_input_fields()
            )
            return execute_with_routing(
                operator, input_record, self.router,
                input_fields_override=all_fields,
            )

        else:  # DOCUMENT
            # Scan all fields once per record; reuse decision for all operators
            key = self._record_key(input_record)
            if key not in self._doc_cache:
                all_fields = (
                    list(operator.input_schema.model_fields)
                    if operator.input_schema is not None
                    else operator.get_input_fields()
                )
                self._doc_cache[key] = self.router.inspect(
                    operator, all_fields, input_record=input_record
                )
            return execute_with_routing(
                operator, input_record, self.router,
                cached_decision=self._doc_cache[key],
            )

    def _execute_plan(
        self,
        plan: PhysicalPlan,
        input_queues: dict[str, dict[str, list]],
        plan_stats: PlanStats,
    ) -> tuple[list[DataRecord], PlanStats]:
        """
        Identical to SequentialSingleThreadExecutionStrategy._execute_plan
        except that the bare `operator(input_record)` call in the else-branch
        is replaced by `execute_with_routing(operator, input_record, self.router)`
        for operators that invoke an LLM.
        """
        output_records = []

        for topo_idx, operator in enumerate(plan):
            source_unique_full_op_ids = (
                [f"source_{operator.get_full_op_id()}"]
                if isinstance(operator, (ContextScanOp, ScanPhysicalOp))
                else plan.get_source_unique_full_op_ids(topo_idx, operator)
            )
            unique_full_op_id = f"{topo_idx}-{operator.get_full_op_id()}"

            num_inputs = sum(
                len(input_queues[unique_full_op_id][src])
                for src in source_unique_full_op_ids
            )
            if num_inputs == 0:
                break

            records, record_op_stats = [], []
            logger.info(f"Processing operator {operator.op_name()} ({unique_full_op_id})")

            # ── Aggregate ────────────────────────────────────────────────────
            if isinstance(operator, AggregateOp):
                source_unique_full_op_id = source_unique_full_op_ids[0]
                record_set = operator(
                    candidates=input_queues[unique_full_op_id][source_unique_full_op_id]
                )
                records = record_set.data_records
                record_op_stats = record_set.record_op_stats
                num_outputs = sum(r._passed_operator for r in records)
                self.progress_manager.incr(
                    unique_full_op_id,
                    num_inputs=1,
                    num_outputs=num_outputs,
                    total_cost=record_set.get_total_cost(),
                )

            # ── Join ─────────────────────────────────────────────────────────
            elif isinstance(operator, JoinOp):
                left_id = source_unique_full_op_ids[0]
                left_n = len(input_queues[unique_full_op_id][left_id])
                left_records = [
                    input_queues[unique_full_op_id][left_id].pop(0)
                    for _ in range(left_n)
                ]
                right_id = source_unique_full_op_ids[1]
                right_n = len(input_queues[unique_full_op_id][right_id])
                right_records = [
                    input_queues[unique_full_op_id][right_id].pop(0)
                    for _ in range(right_n)
                ]
                record_set, num_inputs_processed = operator(left_records, right_records)
                records = record_set.data_records
                record_op_stats = record_set.record_op_stats

                if operator.how in ("left", "right", "outer"):
                    record_set, _ = operator([], [], final=True)
                    records.extend(record_set.data_records)
                    record_op_stats.extend(record_set.record_op_stats)

                num_outputs = sum(r._passed_operator for r in records)
                self.progress_manager.incr(
                    unique_full_op_id,
                    num_inputs=num_inputs_processed,
                    num_outputs=num_outputs,
                    total_cost=record_set.get_total_cost(),
                )

            # ── All other operators (scan, filter, convert, limit, …) ────────
            else:
                source_unique_full_op_id = source_unique_full_op_ids[0]
                for input_record in input_queues[unique_full_op_id][source_unique_full_op_id]:

                    # Privacy hook: route LLM operators; pass others through.
                    # _invoke_operator dispatches to the correct granularity.
                    record_set = self._invoke_operator(operator, input_record)

                    records.extend(record_set.data_records)
                    record_op_stats.extend(record_set.record_op_stats)
                    num_outputs = sum(r._passed_operator for r in record_set.data_records)
                    self.progress_manager.incr(
                        unique_full_op_id,
                        num_inputs=1,
                        num_outputs=num_outputs,
                        total_cost=record_set.get_total_cost(),
                    )

                    if isinstance(operator, LimitScanOp) and len(records) == operator.limit:
                        break

            # ── Update plan stats and pass records to next operator ──────────
            plan_stats.add_record_op_stats(unique_full_op_id, record_op_stats)

            output_records = [r for r in records if r._passed_operator]
            next_id = plan.get_next_unique_full_op_id(topo_idx, operator)
            if next_id is not None:
                input_queues[next_id][unique_full_op_id] = output_records

            logger.info(
                f"Finished processing operator {operator.op_name()} "
                f"({unique_full_op_id}), generated {len(records)} records"
            )

        plan_stats.finish()
        return output_records, plan_stats


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_privacy_processor(
    dataset,
    config: QueryProcessorConfig | None = None,
    router: PrivacyRouter | None = None,
    granularity: RoutingGranularity = RoutingGranularity.OPERATOR,
):
    """
    Build a QueryProcessor whose execution strategy is PrivacyAwareExecutionStrategy.

    This is the primary entry point for end-to-end privacy-routed pipeline runs.
    It calls QueryProcessorFactory.create_processor() normally (handling config
    normalization, model resolution, optimizer setup), then replaces the default
    execution strategy with PrivacyAwareExecutionStrategy, copying execution
    parameters from the already-created strategy so no re-parsing is needed.

    Args:
        dataset: The PZ Dataset (logical plan) to process.
        config:  QueryProcessorConfig; if None, PZ defaults are used.
        router:       PrivacyRouter instance; if None, a default one is created.
        granularity:  RoutingGranularity (OPERATOR/FIELD/DOCUMENT); default OPERATOR.

    Returns:
        QueryProcessor with PrivacyAwareExecutionStrategy installed.

    Example:
        processor = create_privacy_processor(plan, config)
        result = processor.execute()
        print(processor.execution_strategy.router.stats.summary())
    """
    router = router or PrivacyRouter()
    processor = QueryProcessorFactory.create_processor(dataset, config)

    existing = processor.execution_strategy
    privacy_strategy = PrivacyAwareExecutionStrategy(
        router=router,
        granularity=granularity,
        scan_start_idx=getattr(existing, "scan_start_idx", 0),
        max_workers=getattr(existing, "max_workers", 1),
        batch_size=getattr(existing, "batch_size", None),
        num_samples=getattr(existing, "num_samples", None),
        verbose=getattr(existing, "verbose", False),
        progress=getattr(existing, "progress", True),
    )
    processor.execution_strategy = privacy_strategy
    return processor
