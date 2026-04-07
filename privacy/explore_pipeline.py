"""
Palimpzest Enron exploration pipeline.

Loads a small slice of the Enron email testdata, runs a sem_map followed by a
sem_filter using OpenAI GPT-4o, and prints:
  - The logical plan (traversal of the Dataset chain)
  - The physical plan (after optimization, before execution)
  - Input/output fields for each operator
  - The resulting output records

Run from the repo root:
    python privacy/explore_pipeline.py
"""

import os
import sys

# Make sure the repo's src/ is on the path when run from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import palimpzest as pz
from palimpzest.core.lib.schemas import TextFile
from palimpzest.query.processor.query_processor_factory import QueryProcessorFactory


# ---------------------------------------------------------------------------
# 1. Root dataset — first 5 Enron emails (keeps API cost low during exploration)
# ---------------------------------------------------------------------------

class EnronTinyDataset(pz.IterDataset):
    """Loads the first N emails from the Enron eval-medium testdata directory."""

    def __init__(self, data_dir: str, max_emails: int = 5):
        super().__init__(id="enron_tiny", schema=TextFile)
        all_files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".txt")
        )
        self.filepaths = all_files[:max_emails]

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> dict:
        filepath = self.filepaths[idx]
        filename = os.path.basename(filepath)
        with open(filepath, encoding="utf-8", errors="replace") as f:
            contents = f.read()
        return {"filename": filename, "contents": contents}


# ---------------------------------------------------------------------------
# 2. Logical plan construction (lazy — no LLM calls yet)
# ---------------------------------------------------------------------------

def build_plan(data_dir: str) -> pz.Dataset:
    root = EnronTinyDataset(data_dir=data_dir, max_emails=5)

    # sem_map: extract subject and sender from email text (1-to-1)
    plan = root.sem_map(
        cols=[
            {"name": "subject", "type": str, "desc": "The subject line of the email"},
            {"name": "sender",  "type": str, "desc": "The email address of the sender"},
        ],
        desc="Extract email metadata from raw email text.",
        depends_on=["contents"],
    )

    # sem_filter: keep only emails that mention a specific topic
    plan = plan.sem_filter(
        filter="The email discusses scheduling, meetings, or travel",
        depends_on=["contents"],
    )

    return plan


# ---------------------------------------------------------------------------
# 3. Introspection helpers
# ---------------------------------------------------------------------------

def print_logical_plan(dataset: pz.Dataset) -> None:
    """Traverse the Dataset chain and print each logical operator."""
    print("\n" + "=" * 60)
    print("LOGICAL PLAN")
    print("=" * 60)
    for ds in dataset:                         # yields sources first, then self
        op = ds._operator
        in_schema  = op.input_schema
        out_schema = op.output_schema
        in_name  = in_schema.__name__  if in_schema  is not None else "None"
        out_name = out_schema.__name__ if out_schema is not None else "None"
        in_fields  = list(in_schema.model_fields)  if in_schema  is not None else []
        out_fields = list(out_schema.model_fields) if out_schema is not None else []
        new_fields = op.generated_fields

        print(f"\n  [{op.__class__.__name__}]")
        print(f"    input_schema  : {in_name}  {in_fields}")
        print(f"    output_schema : {out_name}  {out_fields}")
        print(f"    generated_fields : {new_fields}")
        if hasattr(op, "depends_on") and op.depends_on:
            print(f"    depends_on    : {op.depends_on}")
        if hasattr(op, "cardinality"):
            print(f"    cardinality   : {op.cardinality}")
        if hasattr(op, "filter") and op.filter is not None:
            print(f"    filter        : {op.filter.get_filter_str()!r}")


def print_physical_plan(plan) -> None:
    """Print the physical plan tree returned by the optimizer."""
    print("\n" + "=" * 60)
    print("PHYSICAL PLAN  (str(plan))")
    print("=" * 60)
    print(str(plan))

    print("\n--- Operator-level field introspection ---")
    for topo_idx, op in enumerate(plan):
        in_fields  = list(op.input_schema.model_fields) if op.input_schema is not None else []
        out_fields = list(op.output_schema.model_fields)
        print(f"\n  [{topo_idx}] {op.op_name()}")
        print(f"       input_fields    : {in_fields}")
        print(f"       output_fields   : {out_fields}")
        print(f"       generated_fields: {op.generated_fields}")
        print(f"       depends_on      : {op.depends_on}")
        model_name = op.get_model_name()
        if model_name:
            print(f"       model           : {model_name}")


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    # Verify OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    # Path to the Enron testdata (relative to repo root)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir  = os.path.join(repo_root, "testdata", "enron-eval-medium")

    if not os.path.isdir(data_dir):
        # Fall back to enron-tiny if medium is not present
        data_dir = os.path.join(repo_root, "testdata", "enron-tiny")

    if not os.path.isdir(data_dir):
        print(f"ERROR: testdata directory not found at {data_dir}")
        sys.exit(1)

    print(f"Using testdata: {data_dir}")

    # Build the lazy logical plan
    plan = build_plan(data_dir)

    # Print the logical plan
    print_logical_plan(plan)

    # ------------------------------------------------------------------
    # Optimization: build a QueryProcessor so we can extract the
    # physical plan before actually executing it.
    #
    # Config notes:
    #   - available_models=[pz.Model.GPT_4o]   → force GPT-4o only
    #   - execution_strategy="sequential"        → simplest; no parallelism
    #   - sentinel_execution_strategy=None       → skip the MAB optimization phase
    #   - progress=False, verbose=False          → keep output clean
    # ------------------------------------------------------------------

    config = pz.QueryProcessorConfig(
        policy=pz.MaxQuality(),
        execution_strategy="sequential",
        sentinel_execution_strategy=None,   # skip optimization / training phase
        available_models=[pz.Model.GPT_4o],
        progress=False,
        verbose=False,
        k=1, j=1, sample_budget=5,
    )

    # Generate unique logical op IDs (normally done inside Dataset.run())
    plan._generate_unique_logical_op_ids()

    # Create a processor — this also calls dataset.relax_types() and
    # resolves model enums.  We use create_processor() (not create_and_run_processor())
    # so we can inspect the physical plan before executing.
    processor = QueryProcessorFactory.create_processor(plan, config)

    # Run the optimizer to get the ranked list of physical plans
    physical_plans = processor.optimizer.optimize(processor.dataset)
    best_plan = physical_plans[0]

    # Print the physical plan
    print_physical_plan(best_plan)

    # ------------------------------------------------------------------
    # Execute: processor.execute() internally re-runs the optimizer and
    # then calls execution_strategy.execute_plan().  We call it here for
    # the full end-to-end run.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXECUTING PLAN  (LLM calls happen here)")
    print("=" * 60)

    result = processor.execute()

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("OUTPUT RECORDS")
    print("=" * 60)
    records = result.data_records
    print(f"  {len(records)} record(s) passed all operators.\n")
    for i, rec in enumerate(records):
        print(f"  Record {i}:")
        for field in rec.__class__.model_fields:
            val = getattr(rec, field, None)
            if isinstance(val, str) and len(val) > 120:
                val = val[:120] + "..."
            print(f"    {field}: {val!r}")
        print()

    # Summary DataFrame if pandas is available
    try:
        df = result.to_df()
        cols = [c for c in df.columns if c not in ("contents",)]
        print("Output DataFrame (key columns):")
        print(df[cols].to_string(index=False))
    except Exception as e:
        print(f"(Could not convert to DataFrame: {e})")


if __name__ == "__main__":
    main()
