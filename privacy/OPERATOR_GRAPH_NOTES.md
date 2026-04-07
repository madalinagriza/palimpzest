# Palimpzest Operator Graph — Architecture Notes
## (Prepared for: Privacy-Aware Routing Integration)

---

## 1. How PZ Constructs Logical and Physical Query Plans

### 1.1 User Program → Logical Plan

Every PZ pipeline starts from a root `Dataset` subclassing one of:
- `pz.IterDataset` — iteration-based data (lists of files, DataFrames, etc.)
- `pz.IndexDataset` — point-lookup / query-based data (vector DBs, SQL)
- `pz.Context` — agent-traversed data (CSV files, time series)

**A `Dataset` object IS the logical plan node.** Each call to a semantic method
(`.sem_filter()`, `.sem_map()`, etc.) constructs a new `LogicalOperator` and wraps it
in a new `Dataset` pointing back at the prior one as its source. The resulting chain
of `Dataset` objects forms a lazy, immutable DAG.

```
user code:
  root  = EnronDataset(...)                  # root Dataset, operator=BaseScan
  step1 = root.sem_map([...])                # Dataset(sources=[root], operator=ConvertScan)
  step2 = step1.sem_filter("...")            # Dataset(sources=[step1], operator=FilteredScan)

logical plan (iterating `step2`):
  BaseScan → ConvertScan → FilteredScan
```

Key files:
- `src/palimpzest/core/data/dataset.py` — `Dataset` class (lines 36-723)
- `src/palimpzest/query/operators/logical.py` — all `LogicalOperator` subclasses

**Iterating** over a `Dataset` yields its sources first, then itself
(`Dataset.__iter__`, line 121), producing a left-to-right topological order. This
is the standard way to inspect the logical plan.

```python
for ds in plan:
    op = ds._operator
    print(op.__class__.__name__, op.input_schema, op.output_schema)
```

### 1.2 Logical Plan → Physical Plan

When the user calls `.run(config)` or `.optimize_and_run(...)`, PZ invokes:

```
Dataset.run()
  └─ QueryProcessorFactory.create_and_run_processor()
       ├─ _config_validation_and_normalization()   # resolves Model enums, strategies
       ├─ dataset.relax_types()                    # makes schemas permissive (Any)
       ├─ Optimizer(cost_model=SampleBasedCostModel())
       └─ processor.execute()
            └─ Optimizer.optimize(dataset)          # returns list[PhysicalPlan], sorted best→worst
                 └─ ImplementationRule.substitute() # for each LogicalOperator,
                                                    # creates one PhysicalOperator per eligible model
```

**Optimizer flow** (`src/palimpzest/query/optimizer/optimizer.py`):

The optimizer is Cascades-style. For each `LogicalOperator` in the plan it:
1. Finds all `ImplementationRule`s whose `matches_pattern()` returns True.
2. Calls `substitute(logical_expression, **runtime_kwargs)` on each rule.
3. Each rule instantiates one `PhysicalOperator` per eligible model in `available_models`.
4. The `CostModel` scores every resulting `PhysicalPlan`.
5. Plans are returned sorted by score (policy-dependent: quality, cost, or time).

The final plan is `plans[0]`. It is a tree of `PhysicalPlan` nodes; iterating over it
(`PhysicalPlan.__iter__`) yields physical operators in topological (left-to-right) order.

**Physical plan string representation:**
```python
str(plan)   # calls plan._get_str() recursively — shows operator + schema info
repr(plan)  # same as str(plan)
```

Key files:
- `src/palimpzest/query/optimizer/optimizer.py` — `Optimizer.optimize()`
- `src/palimpzest/query/optimizer/rules.py` — `ImplementationRule` subclasses (lines 375–1111)
- `src/palimpzest/query/optimizer/plan.py` — `PhysicalPlan` (lines 47–286)
- `src/palimpzest/query/processor/query_processor_factory.py` — factory entry point
- `src/palimpzest/query/processor/query_processor.py` — `QueryProcessor.execute()` (lines 111–153)

---

## 2. Semantic Operators — Full Inventory

All operators are methods on `Dataset`
(`src/palimpzest/core/data/dataset.py`).

### `sem_map` (line 402)
```python
def sem_map(
    self,
    cols: list[dict] | type[BaseModel],   # new fields to compute; each dict has keys
                                           #   'name', 'type', 'desc'
    desc: str | None = None,              # natural-language description of the whole map
    depends_on: str | list[str] | None = None,  # input field(s) the LLM should read
) -> Dataset
```
- **Cardinality**: `ONE_TO_ONE` — one output record per input record.
- **Logical op**: `ConvertScan` (`logical.py` lines 267–306) with `cardinality=ONE_TO_ONE`.
- **Physical ops**: `LLMConvertBonded`, `NonLLMConvert`, `RAGConvert`,
  `CritiqueAndRefineConvert`, `MixtureOfAgents`, `SplitMergeConvert`.
- **Returns**: new `Dataset` whose schema = union(input_schema, new_cols_schema).

### `sem_flat_map` (line 416)
```python
def sem_flat_map(
    self,
    cols: list[dict] | type[BaseModel],
    desc: str | None = None,
    depends_on: str | list[str] | None = None,
) -> Dataset
```
- **Cardinality**: `ONE_TO_MANY` — one input may produce multiple output records.
- **Logical op**: `ConvertScan` with `cardinality=ONE_TO_MANY`.
- Same physical ops as `sem_map`.

### `sem_filter` (line 317)
```python
def sem_filter(
    self,
    filter: str,                           # natural-language predicate
    desc: str | None = None,
    depends_on: str | list[str] | None = None,
) -> Dataset
```
- **Logical op**: `FilteredScan` (`logical.py` lines 343–378).
- **Physical ops**: `LLMFilter` (LLM evaluates predicate), `NonLLMFilter` (UDF-based).
- **Returns**: same schema; records that fail the predicate are marked `_passed_operator=False`.

### `sem_join` (line 269)
```python
def sem_join(
    self,
    other: Dataset,
    condition: str,                        # natural-language join predicate
    desc: str | None = None,
    depends_on: str | list[str] | None = None,
    how: str = "inner",                    # "inner" | "left" | "right" | "outer"
) -> Dataset
```
- **Logical op**: `JoinOp` (`logical.py` lines 415–448).
- **Physical ops**: `NestedLoopsJoin`, `EmbeddingJoin`, `RelationalJoin`.
- **Returns**: `Dataset` with union schema from both inputs.

### `sem_agg` (line 580)
```python
def sem_agg(
    self,
    col: dict | type[BaseModel],           # single output field
    agg: str,                              # natural-language aggregation instruction
    depends_on: str | list[str] | None = None,
) -> Dataset
```
- **Logical op**: `Aggregate` with `agg_str` set (`logical.py` lines 144–198).
- **Physical op**: `SemanticAggregate` (`aggregate.py`).
- **Returns**: `Dataset` with a single aggregated output record.

### `sem_topk` (line 611)
```python
def sem_topk(
    self,
    index: chromadb.Collection,            # ChromaDB vector collection
    search_attr: str,                      # field whose value is the query
    output_attrs: list[dict] | type[BaseModel],  # fields to populate from retrieval
    search_func: Callable | None = None,
    k: int = -1,                           # number of neighbors to retrieve (-1 = all)
) -> Dataset
```
- **Logical op**: `TopKScan` (`logical.py` lines 499–547).
- **Physical op**: `TopKOp` (`topk.py`).
- **Returns**: `Dataset` with `output_attrs` fields populated from retrieved documents.

---

## 3. How PZ Currently Does Model Routing

### Step 1 — Model Discovery (env-var driven)

`src/palimpzest/utils/model_helpers.py`, `get_models()` (line 8):

```python
if os.getenv("OPENAI_API_KEY"):   models += all OpenAI models
if os.getenv("TOGETHER_API_KEY"): models += all Together.AI models
if os.getenv("ANTHROPIC_API_KEY"): models += all Anthropic models
if os.getenv("GEMINI_API_KEY"):   models += all Gemini models
```

### Step 2 — Policy Scoring (optional, lines 78–192)

`get_optimal_models(policy, ...)` scores each model on quality / cost / latency
according to the user's `Policy` (e.g. `MaxQuality`, `MinCost`) and returns the
top `MAX_AVAILABLE_MODELS` models.

### Step 3 — Capability Filtering (per-operator, at optimize time)

**File**: `src/palimpzest/query/optimizer/rules.py`, `ImplementationRule._model_matches_input()` (lines 482–522):

```python
@classmethod
def _model_matches_input(cls, model, logical_expression) -> bool:
    # filters out embedding models
    # checks image/audio field types in input schema
    # returns True only if model capabilities match operator modality
```

### Step 4 — Physical Operator Creation (one per model)

`LLMConvertBondedRule.substitute()` (lines 635–652):
```python
models = [m for m in runtime_kwargs["available_models"]
          if cls._model_matches_input(m, logical_expression)]
for model in models:
    variable_op_kwargs.append({
        "model": model,
        "prompt_strategy": PromptStrategy.MAP,
        "reasoning_effort": runtime_kwargs["reasoning_effort"],
    })
return cls._perform_substitution(logical_expression, LLMConvertBonded, ...)
```
One `PhysicalPlan` is created for every eligible model. The cost model picks the best.

### Step 5 — Plan Selection (cost model)

`src/palimpzest/query/optimizer/cost_model.py` — `SampleBasedCostModel` scores plans.
`Optimizer.optimize()` returns them sorted; `plans[0]` is chosen.

**Summary**: Model selection is fully determined at **optimization time**, not at execution
time. The chosen model is baked into the `PhysicalOperator` instance (as `self.model`).
This is the key constraint for our privacy hook: we must intercept **before or during
optimization**, or override the model on the operator before `__call__` is invoked.

---

## 4. How Operator Inputs / Outputs Are Typed at Runtime

PZ uses **Pydantic `BaseModel`** for all schemas. Every `PhysicalOperator` stores:

```python
self.input_schema   : type[BaseModel]   # schema of records coming IN
self.output_schema  : type[BaseModel]   # schema of records going OUT
self.depends_on     : list[str]         # which input fields this op reads
self.generated_fields : list[str]       # new fields this op adds
```

**Yes — field names and types are fully accessible at runtime** before the LLM call:

```python
# All field names in input schema
list(operator.input_schema.model_fields)        # e.g. ['filename', 'contents']

# All field names that will be generated (new columns)
operator.generated_fields                        # e.g. ['subject', 'sender']

# Fields actually read by this operator
operator.get_input_fields()                      # respects depends_on

# Type annotation for a specific field (a Python type or TypeAlias)
operator.input_schema.model_fields['contents'].annotation   # str

# Description for a specific field
operator.input_schema.model_fields['contents'].description  # "The contents of the file"
```

This means a `PrivacyRouter` can inspect field names like `"email"`, `"name"`,
`"phone"`, `"ssn"` before any LLM call, and route accordingly.

---

## 5. Best Hook Point for Intercepting Before LLM Call

### Option A — Execution Strategy Override (RECOMMENDED)

**File**: `src/palimpzest/query/execution/single_threaded_execution_strategy.py`, line 88:

```python
# inside SequentialSingleThreadExecutionStrategy._execute_plan():
for input_record in input_queues[unique_full_op_id][source_unique_full_op_id]:
    record_set = operator(input_record)   # ← LLM call happens inside __call__
```

**Hook**: Subclass `SequentialSingleThreadExecutionStrategy` (or the parallel variant)
and override `_execute_plan()`. Before calling `operator(input_record)`, inspect the
operator's `input_schema`, `generated_fields`, `get_model_name()`, and optionally
swap `operator.model` for a local model.

This is the cleanest approach because:
- Full operator state is visible (model, schemas, depends_on, generated_fields)
- Full input record is visible (can inspect actual field values for PII scanning)
- No changes to PZ source required — just pass a custom execution strategy

```python
class PrivacyAwareExecutionStrategy(SequentialSingleThreadExecutionStrategy):
    def _execute_plan(self, plan, input_queues, plan_stats):
        for topo_idx, operator in enumerate(plan):
            ...
            for input_record in input_queues[unique_full_op_id][source_unique_full_op_id]:
                # HOOK: inspect before LLM call
                input_fields = operator.get_input_fields()
                chosen_model = privacy_router.route(operator, input_fields)
                if chosen_model == "local" and hasattr(operator, "model"):
                    operator.model = Model(ModelConfig().local_model)
                
                record_set = operator(input_record)
                ...
```

### Option B — Custom `QueryProcessorConfig.execution_strategy`

Pass a custom strategy class via the config. Looking at
`QueryProcessorFactory._create_execution_strategy()` (line 184):
```python
execution_strategy_cls = config.execution_strategy.value   # the class from the Enum
return execution_strategy_cls(**config.to_dict())
```

You can monkey-patch or subclass `ExecutionStrategyType` to include a new entry, or
construct the `ExecutionStrategy` instance directly and pass it to `QueryProcessor`.

### Option C — Physical Operator `__call__` Wrapper

Wrap or monkey-patch `LLMConvertBonded.__call__()` or `LLMFilter.__call__()` before
execution. Less clean — requires reaching into each operator type separately.

### Option D — Optimizer-time Hook

Override `ImplementationRule.substitute()` to filter/modify the model list before
physical operators are created. This affects plan selection, not just execution.
Useful for preventing certain models from ever being considered for sensitive operators.

---

## 6. Where Presidio Will Hook In

**Note**: `presidio-analyzer` and `presidio-anonymizer` are not yet called anywhere in PZ.
The intended integration point is inside the `PrivacyRouter.route()` method
(see `routing_stub.py`), specifically:

```python
from presidio_analyzer import AnalyzerEngine

engine = AnalyzerEngine()
# Called on field values (input_record.<field_name>) or field names
results = engine.analyze(text=field_value, language="en")
# If PII entities found → route to "local"
# Otherwise → route to "cloud"
```

The cleanest trigger is **Option A** (execution strategy hook), where `input_record`
is available with actual data values. This lets Presidio scan real text before it
leaves the process.

Install when ready:
```
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_lg
```

---

## 7. Key File/Line Reference Summary

| What | File | Lines |
|------|------|-------|
| `Dataset` (logical plan node) | `src/palimpzest/core/data/dataset.py` | 36–723 |
| `sem_map` / `sem_flat_map` | `src/palimpzest/core/data/dataset.py` | 402–430 |
| `sem_filter` | `src/palimpzest/core/data/dataset.py` | 317–338 |
| `sem_join` | `src/palimpzest/core/data/dataset.py` | 269–290 |
| `sem_agg` | `src/palimpzest/core/data/dataset.py` | 580–609 |
| `sem_topk` | `src/palimpzest/core/data/dataset.py` | 611–648 |
| `LogicalOperator` base | `src/palimpzest/query/operators/logical.py` | 16–142 |
| `ConvertScan` (sem_map/flat_map) | `src/palimpzest/query/operators/logical.py` | 267–306 |
| `FilteredScan` (sem_filter) | `src/palimpzest/query/operators/logical.py` | 343–378 |
| `Aggregate` (sem_agg) | `src/palimpzest/query/operators/logical.py` | 144–198 |
| `JoinOp` (sem_join) | `src/palimpzest/query/operators/logical.py` | 415–448 |
| `TopKScan` (sem_topk) | `src/palimpzest/query/operators/logical.py` | 499–547 |
| `PhysicalOperator` base | `src/palimpzest/query/operators/physical.py` | 14–226 |
| `PhysicalOperator.get_input_fields()` | `src/palimpzest/query/operators/physical.py` | 172–185 |
| `PhysicalOperator.generated_fields` | `src/palimpzest/query/operators/physical.py` | 60–64 |
| `LLMConvertBonded` | `src/palimpzest/query/operators/convert.py` | 352–372 |
| `LLMFilter` | `src/palimpzest/query/operators/filter.py` | 165–304 |
| `ImplementationRule` base + `_model_matches_input` | `src/palimpzest/query/optimizer/rules.py` | 375–596 |
| `LLMConvertBondedRule.substitute()` | `src/palimpzest/query/optimizer/rules.py` | 635–652 |
| `PhysicalPlan` | `src/palimpzest/query/optimizer/plan.py` | 47–286 |
| `PhysicalPlan.__iter__` (topo order) | `src/palimpzest/query/optimizer/plan.py` | 263–266 |
| `PhysicalPlan.__str__` | `src/palimpzest/query/optimizer/plan.py` | 256–257 |
| Model discovery (`get_models`) | `src/palimpzest/utils/model_helpers.py` | 8–76 |
| Model scoring (`get_optimal_models`) | `src/palimpzest/utils/model_helpers.py` | 78–192 |
| **Main execution loop (HOOK POINT)** | `src/palimpzest/query/execution/single_threaded_execution_strategy.py` | **88** |
| `QueryProcessorConfig` | `src/palimpzest/query/processor/config.py` | 10–63 |
| `QueryProcessorFactory` | `src/palimpzest/query/processor/query_processor_factory.py` | 23–250 |
| Schema field introspection | `src/palimpzest/core/lib/schemas.py` | 60–80 |
