# `depends_on` & `get_input_fields()` Investigation in Palimpzest

## Verdict: The claim is **substantially correct**, with minor clarifications needed.

**Claim checked:** *"Palimpzest operators have a `depends_on` attribute (or metadata field) that declares which input fields the operator reads. `get_input_fields()` respects `depends_on`. This can be used to determine which fields of a record an operator actually accesses, enabling intent-aware routing that skips PII detection on fields the operator doesn't use."*

---

## 1. Does `depends_on` exist?

**Yes.** It is a first-class attribute wired through every layer of the PZ stack:

| Layer | File | Role |
|---|---|---|
| User API | `src/palimpzest/core/data/dataset.py` | Accepted as `str \| list[str] \| None` on `sem_filter()`, `sem_map()`, `sem_flat_map()`, `sem_join()`, `map()`, `flat_map()` |
| Logical operators | `src/palimpzest/query/operators/logical.py` | Stored as `self.depends_on` (sorted list) in `LogicalOperator.__init__()` |
| Physical operators | `src/palimpzest/query/operators/physical.py` | Stored as `self.depends_on` in `PhysicalOperator.__init__()` |

Example user-facing usage (from `demos/enron-demo.py` line 66):
```python
.sem_filter("The email is about ...", depends_on=["contents"])
```

Other real examples in the codebase:

```python
# evals/quest/eval.py line 41
depends_on=["text"]

# abacus-research/biodex-demo.py line 366
plan = plan.sem_map(biodex_ranked_reactions_labels_cols, depends_on=["title", "abstract", "fulltext", "reaction_labels"])

# data/sem_filter_report.md line 20
depends_on=["text", "ssn", "phone", "name"]
```

---

## 2. What does `depends_on` reference?

**Fields/columns of the input schema** — i.e., attribute names declared on a Pydantic `BaseModel` schema class. Values are short field names like `"contents"`, `"title"`, `"abstract"`, `"ssn"`, etc., corresponding to `model_fields` keys on the operator's `input_schema`.

This is visible in `src/palimpzest/query/operators/physical.py` (lines 175–189):

```python
def get_input_fields(self):
    depends_on_fields = (
        [field.split(".")[-1] for field in self.depends_on]
        if self.depends_on is not None and len(self.depends_on) > 0
        else None
    )
    input_fields = (
        list(self.input_schema.model_fields)
        if depends_on_fields is None
        else [field for field in self.input_schema.model_fields if field in depends_on_fields]
    )
    return input_fields
```

The `field.split(".")[-1]` handles a potential `{dataset_id}.{field_name}` format, but in practice users pass bare field names.

---

## 3. What is the data model?

PZ operates on **schema-typed structured records**. Each record is a `DataRecord` wrapping a Pydantic `BaseModel` instance (`src/palimpzest/core/elements/records.py` lines 28–100+):

```python
class DataRecord:
    def __init__(self, data_item: BaseModel, ...):
        self._data_item = data_item

    def get_field_names(self):
        return list(type(self._data_item).model_fields.keys())

    def get_field_type(self, field_name: str) -> FieldInfo:
        return type(self._data_item).model_fields[field_name]
```

Records are **not** opaque blobs. They have named, typed fields accessible individually. Operators receive a `DataRecord` and a list of field names to work with.

---

## 4. Does `get_input_fields()` exist and respect `depends_on`?

**Yes, exactly.** `PhysicalOperator.get_input_fields()` at `src/palimpzest/query/operators/physical.py` (lines 175–189):

- If `depends_on` is `None` or empty → returns **all** `input_schema.model_fields`
- If `depends_on` is set → returns **only** the listed fields (filtered to those present in `input_schema`)

This result is consumed downstream as `project_cols`:

### Filter operators (`src/palimpzest/query/operators/filter.py`, line 249)

```python
def filter(self, candidate: DataRecord) -> tuple[dict[str, bool], GenerationStats]:
    # get the set of input fields to use for the filter operation
    input_fields = self.get_input_fields()

    # construct kwargs for generation
    gen_kwargs = {"project_cols": input_fields, "filter_condition": self.filter_obj.filter_condition}

    fields = {"passed_operator": FieldInfo(annotation=bool, description="Whether the record passed the filter operation")}
    field_answers, _, generation_stats, _ = self.generator(candidate, fields, **gen_kwargs)

    return field_answers, generation_stats
```

### Convert operators (`src/palimpzest/query/operators/convert.py`, line 356)

```python
def convert(self, candidate: DataRecord, fields: dict[str, FieldInfo]) -> tuple[dict[str, list], GenerationStats]:
    # get the set of input fields to use for the convert operation
    input_fields = self.get_input_fields()

    # construct kwargs for generation
    gen_kwargs = {"project_cols": input_fields, "output_schema": self.output_schema}

    field_answers, _, generation_stats, _ = self.generator(candidate, fields, **gen_kwargs)
    ...
    return field_answers, generation_stats
```

### PromptFactory (`src/palimpzest/prompts/prompt_factory.py`, line 413)

```python
def _get_input_fields(self, candidate: DataRecord, **kwargs) -> list[str]:
    # If project_cols is provided, use it; otherwise fall back to all candidate fields
    input_fields = kwargs.get("project_cols", candidate.get_field_names())
    input_fields = [field for field in input_fields if field in candidate.get_field_names()]
    return input_fields
```

**The chain is:**
```
depends_on (set at plan time)
  → PhysicalOperator.get_input_fields()
    → project_cols kwarg
      → PromptFactory._get_input_fields()
        → only those fields appear in the LLM prompt
```

---

## 5. The full flow: User API → Prompt

### User API (`src/palimpzest/core/data/dataset.py`, line 317)

```python
def sem_filter(
    self,
    filter: str,
    desc: str | None = None,
    depends_on: str | list[str] | None = None,
) -> Dataset:
    f = Filter(filter)

    if isinstance(depends_on, str):
        depends_on = [depends_on]

    operator = FilteredScan(input_schema=self.schema, output_schema=self.schema, filter=f, desc=desc, depends_on=depends_on)
    return Dataset(sources=[self], operator=operator, schema=self.schema)
```

### Logical operator (`src/palimpzest/query/operators/logical.py`, lines 37–53)

```python
class LogicalOperator:
    def __init__(
        self,
        output_schema: type[BaseModel],
        input_schema: type[BaseModel] | None = None,
        depends_on: list[str] | None = None,
    ):
        self.depends_on = [] if depends_on is None else sorted(depends_on)
        # ...
```

### Physical operator (`src/palimpzest/query/operators/physical.py`, lines 20–64)

```python
def __init__(self, ..., depends_on: list[str] | None = None, ...):
    self.depends_on = depends_on if depends_on is None else sorted(depends_on)

    # Also used to compute input modalities:
    depends_on_short_field_names = [field.split(".")[-1] for field in self.depends_on] if self.depends_on is not None else None
    self.input_modalities = None
    if self.input_schema is not None:
        self.input_modalities = set()
        for field_name, field in self.input_schema.model_fields.items():
            if self.depends_on is not None and field_name not in depends_on_short_field_names:
                continue
            # ... assign modality (IMAGE, AUDIO, TEXT)
```

---

## 6. Known gaps / caveats

### `_resolve_depends_on()` is unimplemented (`src/palimpzest/core/data/dataset.py`, line 178)

```python
def _resolve_depends_on(self, depends_on: list[str]) -> list[str]:
    """
    TODO: resolve the `depends_on` strings to their full field names ({Dataset.id}.{field_name}).
    """
    return []
```

This is a stub for future fully-qualified field name resolution. It does not block the current mechanism — users simply pass bare field names and they work.

### `depends_on` is optional

If the user omits `depends_on`, `get_input_fields()` returns **all** input schema fields. A PII router must treat "no `depends_on`" as "reads all fields" (conservative default).

---

## 7. Summary of claim accuracy

| Sub-claim | Accurate? | Notes |
|---|---|---|
| `depends_on` attribute exists | **Yes** | On both `LogicalOperator` and `PhysicalOperator` |
| It declares which input fields the operator reads | **Yes** | Field names matching `input_schema.model_fields` keys |
| `get_input_fields()` respects `depends_on` | **Yes** | Returns only `depends_on` fields when set; all fields when not |
| Can determine which fields an operator accesses at routing time | **Yes** | `depends_on` is set at plan construction time; queryable before execution |
| Enables intent-aware routing that skips PII on unused fields | **Feasible** | Field-level granularity is real; only `depends_on` fields are sent to the LLM |
| Operators always see the full document | **No (when `depends_on` is set)** | Only projected fields enter the prompt |
