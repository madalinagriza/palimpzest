"""
Privacy-aware routing stub for Palimpzest.

Provides:
  - ModelConfig       — local vs. cloud model identifiers
  - RoutingStats      — per-run aggregate counters for benchmarking
  - PrivacyRouter     — decides "local" or "cloud" per operator
  - execute_with_routing — wraps a single operator call with routing logic

This module is fully importable without runtime errors and requires no
modifications to PZ source files in src/palimpzest/.

Current behavior:
  - Tries to use Presidio if installed (module-level singleton; init cost paid once).
  - Falls back to lightweight regex / field-name heuristics if Presidio is not
    available.
  - Logs which fields and entity types triggered the decision.
  - Records aggregate routing stats on router.stats for benchmark reporting.
  - Swaps operator.model to the local Model instance when routing to "local",
    handling both plain-string models and PZ Model instances.
"""

from __future__ import annotations

import sys
import os
import re
import threading

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Ensure the PZ src is importable when this module is imported directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Routing granularity
# ---------------------------------------------------------------------------

class RoutingGranularity(str, Enum):
    """
    Controls how broadly PII is scanned before routing a single operator call.

    OPERATOR  — scan only the fields the operator actually reads (get_input_fields(),
                which respects depends_on). Most precise; default.
    FIELD     — scan all fields present in the operator's input_schema, ignoring
                depends_on. Routes to local if any input field has PII, even if
                this operator doesn't need it.
    DOCUMENT  — scan all fields of the raw record once per document; cache the
                local/cloud decision and reuse it for every operator that touches
                that record. Coarsest: one PII hit anywhere → whole pipeline local.
    """
    OPERATOR = "operator"
    FIELD    = "field"
    DOCUMENT = "document"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Maps the two routing destinations to actual model identifiers.

    local_model        — model tag served locally via Ollama (no data leaves the machine)
    local_api_base     — Ollama server base URL
    cloud_model        — served via OpenAI API (highest quality, data sent externally)
    score_threshold    — minimum Presidio confidence score to count as a detection
    sensitive_entities — the only Presidio entity types treated as routing-relevant PII;
                         noisy resume entities (DATE_TIME, LOCATION, PERSON, NRP) are
                         intentionally excluded because they produce too many false positives
    """
    local_model: str = "ollama/llama3.1:8b"
    local_api_base: str = "http://localhost:11434"
    cloud_model: str = "gpt-4o"
    score_threshold: float = 0.6
    sensitive_entities: frozenset = frozenset({
        "US_SSN",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "US_DRIVER_LICENSE",
        "IP_ADDRESS",
        "US_BANK_NUMBER",
        "US_PASSPORT",
    })


# ---------------------------------------------------------------------------
# Detection and routing result types
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single PII detection hit used for logging / debugging."""

    field_name: str
    entity_type: str
    source: str
    preview: str = ""
    score: float | None = None


@dataclass
class RouteDecision:
    """Structured routing decision for one operator invocation."""

    destination: str
    detections: list[Detection] = field(default_factory=list)
    inspected_fields: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class RoutingStats:
    """
    Aggregate routing counters accumulated across all operator calls in a
    pipeline run. Attach one instance to a PrivacyRouter and read it after
    processor.execute() to get the numbers needed for the benchmark table.
    """

    total: int = 0
    routed_local: int = 0
    routed_cloud: int = 0
    detections_by_entity: dict[str, int] = field(default_factory=dict)

    def record(self, decision: RouteDecision) -> None:
        """Update counters from a single RouteDecision."""
        self.total += 1
        if decision.destination == "local":
            self.routed_local += 1
        else:
            self.routed_cloud += 1
        for d in decision.detections:
            self.detections_by_entity[d.entity_type] = (
                self.detections_by_entity.get(d.entity_type, 0) + 1
            )

    def summary(self) -> str:
        """One-line summary suitable for logging or a results table."""
        if self.total == 0:
            return "no operators processed"
        pct = 100.0 * self.routed_local / self.total
        top = sorted(self.detections_by_entity.items(), key=lambda x: -x[1])[:5]
        return (
            f"total={self.total}  local={self.routed_local} ({pct:.1f}%)  "
            f"cloud={self.routed_cloud}  top_entities={top}"
        )


# ---------------------------------------------------------------------------
# Module-level Presidio singleton
# Constructing AnalyzerEngine takes ~1-2 s (spaCy model load).  We pay that
# cost exactly once per process, not once per PrivacyRouter instance.
# ---------------------------------------------------------------------------

_ANALYZER_ENGINE = None
_ANALYZER_LOCK = threading.Lock()


def _get_shared_analyzer():
    """Return the shared AnalyzerEngine, initializing it on first call."""
    global _ANALYZER_ENGINE
    if _ANALYZER_ENGINE is not None:
        return _ANALYZER_ENGINE
    with _ANALYZER_LOCK:
        if _ANALYZER_ENGINE is not None:   # double-checked inside lock
            return _ANALYZER_ENGINE
        try:
            from presidio_analyzer import AnalyzerEngine
            _ANALYZER_ENGINE = AnalyzerEngine()
        except Exception:
            _ANALYZER_ENGINE = None        # Presidio not installed — use fallback
    return _ANALYZER_ENGINE


# ---------------------------------------------------------------------------
# Privacy Router
# ---------------------------------------------------------------------------

class PrivacyRouter:
    """
    Decides whether a given operator should run against a local or cloud model.

    Routing policy (v1):
      - inspect the actual input values of the fields the operator reads
        (operator.get_input_fields() already respects depends_on)
      - if any field appears to contain PII -> route to "local"
      - otherwise -> route to "cloud"

    The router prefers Presidio when available, but still works without it using
    field-name and regex heuristics.  All routing decisions are tallied in
    self.stats for post-run benchmark reporting.
    """

    _SENSITIVE_FIELD_HINTS = {
        "name",
        "email",
        "phone",
        "ssn",
        "social_security",
        "address",
        "ip",
        "ip_address",
        "credit_card",
        "card_number",
        "dob",
        "birthdate",
        "birth_date",
        "person",
        "patient",
    }

    _REGEX_PATTERNS: dict[str, re.Pattern[str]] = {
        "EMAIL_ADDRESS": re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b"),
        "PHONE_NUMBER": re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"),
        "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    }

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.last_decision: RouteDecision | None = None
        self.stats = RoutingStats()

    @staticmethod
    def _safe_preview(value: str, limit: int = 60) -> str:
        """Return a compact preview for logging without dumping full records."""
        cleaned = " ".join(value.split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 3] + "..."

    @staticmethod
    def _coerce_text(value: Any) -> str:
        """Convert an arbitrary field value into a string for scanning."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _detect_with_presidio(self, field_name: str, text: str) -> list[Detection]:
        """
        Run Presidio on a field value, returning only detections that are:
          (a) in config.sensitive_entities — avoids noisy resume entities like
              DATE_TIME, LOCATION, PERSON, NRP that cause false positives, and
          (b) meet config.score_threshold — filters low-confidence matches.
        """
        analyzer = _get_shared_analyzer()
        if analyzer is None or not text.strip():
            return []

        try:
            results = analyzer.analyze(text=text, language="en")
        except Exception:
            return []

        detections: list[Detection] = []
        for result in results:
            entity_type = getattr(result, "entity_type", "PII")
            score = getattr(result, "score", 0.0) or 0.0

            # Skip entity types that are too noisy on resume text
            if entity_type not in self.config.sensitive_entities:
                continue
            # Skip low-confidence detections
            if score < self.config.score_threshold:
                continue

            start = max(0, getattr(result, "start", 0))
            end = min(len(text), getattr(result, "end", 0))
            snippet = text[start:end] if end > start else text[:40]
            detections.append(
                Detection(
                    field_name=field_name,
                    entity_type=entity_type,
                    source="presidio",
                    score=score,
                    preview=self._safe_preview(snippet),
                )
            )
        return detections

    def _detect_with_heuristics(self, field_name: str, text: str) -> list[Detection]:
        """
        Fallback detector using field-name hints and regexes.

        Field-name hints only fire when the field has a non-empty value —
        this avoids false positives from schema fields like `ssn` or `phone`
        that are present but empty in the record.
        """
        detections: list[Detection] = []

        if not text.strip():
            return detections  # empty field — nothing to detect

        lowered_name = field_name.lower()
        field_tokens = set(re.split(r"[^a-z0-9]+", lowered_name)) - {""}

        for hint in self._SENSITIVE_FIELD_HINTS:
            if hint in field_tokens:
                detections.append(
                    Detection(
                        field_name=field_name,
                        entity_type=f"FIELD_HINT:{hint.upper()}",
                        source="heuristic",
                        preview=self._safe_preview(text),
                    )
                )
                break

        for entity_type, pattern in self._REGEX_PATTERNS.items():
            match = pattern.search(text)
            if match is None:
                continue
            detections.append(
                Detection(
                    field_name=field_name,
                    entity_type=entity_type,
                    source="regex",
                    preview=self._safe_preview(match.group(0)),
                )
            )

        return detections

    def inspect(self, operator, input_fields: list[str], input_record: Any | None = None) -> RouteDecision:
        """
        Inspect the fields an operator will read and build a routing decision.

        get_input_fields() on a real PZ operator already respects depends_on,
        so only the fields this operator actually reads are scanned.

        Args:
            operator:     PZ PhysicalOperator-like object.
            input_fields: List of field names the operator reads.
            input_record: Concrete record whose values can be scanned.

        Returns:
            RouteDecision with destination + detailed detections.
        """
        detections: list[Detection] = []
        inspected_fields = list(input_fields)

        for field_name in input_fields:
            value = getattr(input_record, field_name, "") if input_record is not None else ""
            text = self._coerce_text(value)

            field_detections = self._detect_with_presidio(field_name, text)
            if not field_detections:
                field_detections = self._detect_with_heuristics(field_name, text)

            detections.extend(field_detections)

        if detections:
            decision = RouteDecision(
                destination="local",
                detections=detections,
                inspected_fields=inspected_fields,
                reason="detected sensitive data in operator inputs",
            )
        else:
            decision = RouteDecision(
                destination="cloud",
                detections=[],
                inspected_fields=inspected_fields,
                reason="no sensitive data detected in operator inputs",
            )

        self.stats.record(decision)
        return decision

    def route(self, operator, input_fields: list[str], input_record: Any | None = None) -> str:
        """Compatibility wrapper returning only the destination string."""
        decision = self.inspect(operator, input_fields, input_record=input_record)
        self.last_decision = decision
        return decision.destination


# ---------------------------------------------------------------------------
# Model swap helper
# ---------------------------------------------------------------------------

def _set_operator_model_if_possible(
    operator,
    chosen_model: str,
    api_base: str = "http://localhost:11434",
) -> bool:
    """
    Best-effort model override.

    Handles two cases:
      1. operator.model is a plain string  — overwrite directly.
      2. operator.model is a PZ Model instance — construct a new Model via
         the vLLM/api_base path so PZ's generator can reach the local server.

    Returns True if the swap succeeded, False otherwise.
    """
    if not hasattr(operator, "model"):
        return False

    current_model = getattr(operator, "model")

    # Case 1: plain string
    if isinstance(current_model, str):
        setattr(operator, "model", chosen_model)
        return True

    # Case 2: PZ Model instance — use the vLLM constructor path
    try:
        from palimpzest.constants import Model as PZModel
        if isinstance(current_model, PZModel):
            new_model = PZModel(chosen_model, api_base=api_base)
            setattr(operator, "model", new_model)
            return True
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# Execution wrapper
# ---------------------------------------------------------------------------

def execute_with_routing(
    operator,
    input_record,
    router: PrivacyRouter,
    *,
    input_fields_override: list[str] | None = None,
    cached_decision: RouteDecision | None = None,
):
    """
    Wrap a single operator execution with privacy-aware model routing.

    Steps:
      1. If operator has no model (scan, limit) — pass through directly.
      2. Determine input fields to scan (override → all schema fields → depends_on).
      3. Build a RouteDecision: use cached_decision if provided (document-level),
         otherwise call router.inspect() (records stats automatically).
      4. Attempt to swap operator.model to local or cloud model.
      5. Log the full decision.
      6. Execute operator(input_record) and return the result.

    Args:
        operator:              A PZ PhysicalOperator instance.
        input_record:          A pz DataRecord to pass to the operator.
        router:                A PrivacyRouter instance.
        input_fields_override: If set, scan these fields instead of get_input_fields().
                               Used for FIELD-level granularity.
        cached_decision:       If set, skip detection and use this pre-computed decision.
                               Used for DOCUMENT-level granularity (subsequent operators).
                               Stats are still recorded for every operator call.

    Returns:
        DataRecordSet — the output from operator(input_record).
    """
    # Non-LLM operators (scans, limits) have no model — pass through.
    if operator.get_model_name() is None:
        return operator(input_record)

    generated_fields: list[str] = operator.generated_fields
    op_name: str = operator.op_name()
    model_name = operator.get_model_name()

    if cached_decision is not None:
        # Document-level: reuse a pre-scanned decision, but still tally stats
        # for this operator call so the benchmark counts are per-operator.
        decision = RouteDecision(
            destination=cached_decision.destination,
            detections=cached_decision.detections,
            inspected_fields=cached_decision.inspected_fields,
            reason=f"document-level cached: {cached_decision.reason}",
        )
        router.stats.record(decision)
        router.last_decision = decision
    else:
        input_fields: list[str] = (
            input_fields_override
            if input_fields_override is not None
            else operator.get_input_fields()
        )
        decision = router.inspect(operator, input_fields, input_record=input_record)
        router.last_decision = decision

    destination = decision.destination
    input_fields = decision.inspected_fields

    chosen_model = (
        router.config.cloud_model if destination == "cloud"
        else router.config.local_model
    )
    model_swapped = _set_operator_model_if_possible(
        operator, chosen_model, api_base=router.config.local_api_base
    )

    detection_summary = [
        {
            "field": d.field_name,
            "entity": d.entity_type,
            "source": d.source,
            "preview": d.preview,
            "score": d.score,
        }
        for d in decision.detections
    ]

    print(
        f"[PrivacyRouter] {op_name}"
        f"  |  input_fields={input_fields}"
        f"  |  generates={generated_fields}"
        f"  |  current_model={model_name}"
        f"  |  routed_to={destination!r} ({chosen_model})"
        f"  |  model_swapped={model_swapped}"
        f"  |  reason={decision.reason!r}"
        f"  |  detections={detection_summary}"
    )

    record_set = operator(input_record)
    return record_set


# ---------------------------------------------------------------------------
# Quick smoke-test (run this file directly to verify importability)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = ModelConfig()
    print(f"ModelConfig: local={config.local_model!r}  api_base={config.local_api_base!r}  cloud={config.cloud_model!r}")

    router = PrivacyRouter(config)

    class _FakeOperator:
        generated_fields = ["subject", "sender"]
        model = "gpt-4o"

        def op_name(self):
            return "LLMConvertBonded"

        def get_input_fields(self):
            return ["filename", "contents"]

        def get_model_name(self):
            return "openai/gpt-4o-2024-08-06"

        def __call__(self, record):
            print("Operator executed.")
            return None

    class _FakeRecord:
        filename = "email1.txt"
        contents = "Contact John Smith at john@example.com or 617-555-1212."

    fake_op = _FakeOperator()
    fake_record = _FakeRecord()

    decision = router.inspect(fake_op, fake_op.get_input_fields(), input_record=fake_record)
    print(f"Routing decision for {fake_op.op_name()!r}: {decision.destination!r}")
    assert decision.destination == "local", "Expected 'local' because test record contains obvious PII"

    # Verify stats are recorded
    assert router.stats.total == 1
    assert router.stats.routed_local == 1
    assert router.stats.routed_cloud == 0
    print(f"Stats after one call: {router.stats.summary()}")

    # Verify model swap works for plain-string model
    swapped = _set_operator_model_if_possible(fake_op, config.local_model, api_base=config.local_api_base)
    assert swapped, "Expected model swap to succeed for plain-string model"
    assert fake_op.model == config.local_model

    execute_with_routing(fake_op, fake_record, router)
    print("routing_stub.py: all checks passed.")
