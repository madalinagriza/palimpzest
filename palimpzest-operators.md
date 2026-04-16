# Palimpzest Operators

This document summarizes the operators found by searching the repository codebase.

## Public API Operators

Defined on `Dataset` in `src/palimpzest/core/data/dataset.py`.

### Semantic operators

- `sem_join`
- `sem_filter`
- `sem_map`
- `sem_flat_map`
- `sem_agg`
- `sem_topk`

### Relational and UDF operators

- `join`
- `filter`
- `map`
- `flat_map`
- `groupby`
- `project`
- `limit`
- `distinct`

### Built-in aggregate operators

- `count`
- `average`
- `sum`
- `min`
- `max`

### Deprecated aliases still present

- `sem_add_columns`
- `add_columns`

### Context API operator-like entrypoints

Defined on `Context` in `src/palimpzest/core/data/context.py`.

- `compute`
- `search`

## Logical Operators

Defined in `src/palimpzest/query/operators/logical.py`.

- `LogicalOperator`
- `Aggregate`
- `BaseScan`
- `ContextScan`
- `ConvertScan`
- `Distinct`
- `FilteredScan`
- `GroupByAggregate`
- `JoinOp`
- `LimitScan`
- `Project`
- `TopKScan`
- `ComputeOperator`
- `SearchOperator`

## Exported Logical Operator Registry

Defined in `src/palimpzest/query/operators/__init__.py` as `LOGICAL_OPERATORS`.

- `LogicalOperator`
- `Aggregate`
- `BaseScan`
- `ConvertScan`
- `Distinct`
- `FilteredScan`
- `GroupByAggregate`
- `JoinOp`
- `LimitScan`
- `Project`
- `TopKScan`

### Defined but not included in `LOGICAL_OPERATORS`

- `ContextScan`
- `ComputeOperator`
- `SearchOperator`

## Physical Operators

Defined across `src/palimpzest/query/operators/`.

### Base and abstract physical operators

- `PhysicalOperator`
- `AggregateOp`
- `ConvertOp`
- `FilterOp`
- `JoinOp`
- `ScanPhysicalOp`

### Aggregate physical operators

- `ApplyGroupByOp`
- `AverageAggregateOp`
- `CountAggregateOp`
- `MinAggregateOp`
- `MaxAggregateOp`
- `SumAggregateOp`
- `SemanticAggregate`

### Convert physical operators

- `NonLLMConvert`
- `LLMConvert`
- `LLMConvertBonded`

### Filter physical operators

- `NonLLMFilter`
- `LLMFilter`

### Scan and context physical operators

- `MarshalAndScanDataOp`
- `ContextScanOp`

### Join physical operators

- `RelationalJoin`
- `LLMJoin`
- `NestedLoopsJoin`
- `EmbeddingJoin`

### Other semantic implementation variants

- `RAGConvert`
- `RAGFilter`
- `MixtureOfAgentsConvert`
- `MixtureOfAgentsFilter`
- `CritiqueAndRefineConvert`
- `CritiqueAndRefineFilter`
- `SplitConvert`
- `SplitFilter`

### Other physical operators

- `DistinctOp`
- `ProjectOp`
- `LimitScanOp`
- `TopKOp`

### Agent/context physical operators

- `SmolAgentsCompute`
- `SmolAgentsSearch`

## Exported Physical Operator Registry

Defined in `src/palimpzest/query/operators/__init__.py` as `PHYSICAL_OPERATORS`.

- `AggregateOp`
- `ApplyGroupByOp`
- `AverageAggregateOp`
- `CountAggregateOp`
- `MaxAggregateOp`
- `MinAggregateOp`
- `SemanticAggregate`
- `SumAggregateOp`
- `ConvertOp`
- `NonLLMConvert`
- `LLMConvert`
- `LLMConvertBonded`
- `CritiqueAndRefineConvert`
- `CritiqueAndRefineFilter`
- `DistinctOp`
- `ScanPhysicalOp`
- `MarshalAndScanDataOp`
- `FilterOp`
- `NonLLMFilter`
- `LLMFilter`
- `EmbeddingJoin`
- `JoinOp`
- `NestedLoopsJoin`
- `LimitScanOp`
- `MixtureOfAgentsConvert`
- `MixtureOfAgentsFilter`
- `PhysicalOperator`
- `ProjectOp`
- `RAGConvert`
- `RAGFilter`
- `TopKOp`
- `SplitConvert`
- `SplitFilter`

### Defined but not included in `PHYSICAL_OPERATORS`

- `ContextScanOp`
- `RelationalJoin`
- `LLMJoin`
- `SmolAgentsCompute`
- `SmolAgentsSearch`

## Optimizer Wiring Notes

Defined in `src/palimpzest/query/optimizer/rules.py`.

### Basic substitution map

- `BaseScan -> MarshalAndScanDataOp`
- `SearchOperator -> SmolAgentsSearch`
- `ContextScan -> ContextScanOp`
- `Distinct -> DistinctOp`
- `LimitScan -> LimitScanOp`
- `Project -> ProjectOp`
- `GroupByAggregate -> ApplyGroupByOp`

### Special-case compute rule

`ComputeOperator` is not in the basic substitution map. Instead, it is handled by `AddContextsBeforeComputeRule`, which substitutes it with `SmolAgentsCompute` after retrieving additional context.

### Implementation-rule-based operator families

These logical operators can map to multiple physical implementations depending on optimizer rules:

- `ConvertScan`
  - `NonLLMConvert`
  - `LLMConvert`
  - `LLMConvertBonded`
  - `RAGConvert`
  - `MixtureOfAgentsConvert`
  - `CritiqueAndRefineConvert`
  - `SplitConvert`

- `FilteredScan`
  - `NonLLMFilter`
  - `LLMFilter`
  - `RAGFilter`
  - `MixtureOfAgentsFilter`
  - `CritiqueAndRefineFilter`
  - `SplitFilter`

- `JoinOp`
  - `RelationalJoin`
  - `NestedLoopsJoin`
  - `EmbeddingJoin`

- `Aggregate`
  - `CountAggregateOp`
  - `AverageAggregateOp`
  - `SumAggregateOp`
  - `MinAggregateOp`
  - `MaxAggregateOp`
  - `SemanticAggregate`

- `TopKScan`
  - `TopKOp`

## Source Files

- `src/palimpzest/core/data/dataset.py`
- `src/palimpzest/core/data/context.py`
- `src/palimpzest/query/operators/__init__.py`
- `src/palimpzest/query/operators/logical.py`
- `src/palimpzest/query/operators/aggregate.py`
- `src/palimpzest/query/operators/convert.py`
- `src/palimpzest/query/operators/filter.py`
- `src/palimpzest/query/operators/join.py`
- `src/palimpzest/query/operators/scan.py`
- `src/palimpzest/query/operators/compute.py`
- `src/palimpzest/query/operators/search.py`
- `src/palimpzest/query/operators/rag.py`
- `src/palimpzest/query/operators/mixture_of_agents.py`
- `src/palimpzest/query/operators/critique_and_refine.py`
- `src/palimpzest/query/operators/split.py`
- `src/palimpzest/query/operators/topk.py`
- `src/palimpzest/query/operators/project.py`
- `src/palimpzest/query/operators/distinct.py`
- `src/palimpzest/query/operators/limit.py`
- `src/palimpzest/query/optimizer/rules.py`
