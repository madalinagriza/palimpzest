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
  - route "local" when sensitive data is detected and the operator prompt appears to need it
  - route "cloud_anonymized" when sensitive data is detected but the prompt does not appear to need it
  - route "cloud" when no sensitive data is detected

This file remains importable without Presidio installed: it falls back to regex and
field-name heuristics. If Presidio is installed, it uses AnalyzerEngine and
AnonymizerEngine.
"""

from __future__ import annotations

import sys
import copy
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
    sensitive_entities: frozenset[str] = frozenset({
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
    query_text: str = ""
    query_needs_sensitive: bool | None = None


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
    routed_cloud_anonymized: int = 0
    detections_by_entity: dict[str, int] = field(default_factory=dict)

    def record(self, decision: RouteDecision) -> None:
        """Update counters from a single RouteDecision."""
        self.total += 1
        if decision.destination == "local":
            self.routed_local += 1
        elif decision.destination == "cloud_anonymized":
            self.routed_cloud_anonymized += 1
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
            f"cloud={self.routed_cloud}  cloud_anonymized={self.routed_cloud_anonymized}  "
            f"top_entities={top}"
        )


# ---------------------------------------------------------------------------
# Module-level Presidio singleton
# Constructing AnalyzerEngine takes ~1-2 s (spaCy model load).  We pay that
# cost exactly once per process, not once per PrivacyRouter instance.
# ---------------------------------------------------------------------------

_ANALYZER_ENGINE = None
_ANALYZER_LOCK = threading.Lock()
_ANONYMIZER_ENGINE = None
_ANONYMIZER_LOCK = threading.Lock()


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


def _get_shared_anonymizer():
    global _ANONYMIZER_ENGINE
    if _ANONYMIZER_ENGINE is not None:
        return _ANONYMIZER_ENGINE
    with _ANONYMIZER_LOCK:
        if _ANONYMIZER_ENGINE is not None:
            return _ANONYMIZER_ENGINE
        try:
            from presidio_anonymizer import AnonymizerEngine
            _ANONYMIZER_ENGINE = AnonymizerEngine()
        except Exception:
            _ANONYMIZER_ENGINE = None
    return _ANONYMIZER_ENGINE


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
        "name", "email", "phone", "ssn", "social_security", "address", "ip",
        "ip_address", "credit_card", "card_number", "dob", "birthdate",
        "birth_date", "person", "patient",
    }

    _REGEX_PATTERNS: dict[str, re.Pattern[str]] = {
        "EMAIL_ADDRESS": re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b"),
        "PHONE_NUMBER": re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"),
        "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    }

    _SENSITIVE_QUERY_KEYWORDS = {
        "pii", "personally identifiable", "sensitive", "ssn", "social security",
        "phone number", "email address", "mail address", "home address", "credit card", "passport", "driver",
        "person's name", "full name", "sender", "recipient", "contact info", "identity",
        "patient", "birth", "dob", "ip address",
    }

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.last_decision: RouteDecision | None = None
        self.stats = RoutingStats()

    @staticmethod
    def _safe_preview(value: str, limit: int = 60) -> str:
        """Return a compact preview for logging without dumping full records."""
        cleaned = " ".join(value.split())
        return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."

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

    def _extract_query_text(self, operator) -> str:
        chunks: list[str] = []

        filt = getattr(operator, "filter", None)
        if filt is not None:
            try:
                chunks.append(filt.get_filter_str())
            except Exception:
                chunks.append(str(filt))

        for attr in ("desc", "description", "agg_str", "condition"):
            value = getattr(operator, attr, None)
            if value:
                chunks.append(str(value))

        generated = getattr(operator, "generated_fields", []) or []
        chunks.extend(str(f) for f in generated)

        output_schema = getattr(operator, "output_schema", None)
        if output_schema is not None:
            for field_name in generated:
                try:
                    field_info = output_schema.model_fields.get(field_name)
                    if field_info is not None:
                        chunks.append(field_name)
                        desc = getattr(field_info, "description", None)
                        if desc:
                            chunks.append(str(desc))
                except Exception:
                    pass

        return " ".join(c for c in chunks if c).strip()

    def _query_needs_sensitive_data(self, query_text: str, detections: list[Detection]) -> bool:
        # Conservative fallback: if we cannot see the operator prompt, keep data local.
        if not query_text.strip():
            return True

        q = query_text.lower()
        if any(keyword in q for keyword in self._SENSITIVE_QUERY_KEYWORDS):
            return True

        for d in detections:
            field = d.field_name.lower()
            entity = d.entity_type.lower()
            if field and f" {field} " in f" {q} ":
                return True
            # Avoid matching generic entity tokens like "email" from EMAIL_ADDRESS
            # against prompts like "email discusses scheduling".
        return False

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

        query_text = self._extract_query_text(operator)

        if detections:
            needs_sensitive = self._query_needs_sensitive_data(query_text, detections)
            if needs_sensitive:
                decision = RouteDecision(
                    destination="local",
                    detections=detections,
                    inspected_fields=inspected_fields,
                    reason="detected sensitive data and operator prompt appears to need it",
                    query_text=query_text,
                    query_needs_sensitive=True,
                )
            else:
                decision = RouteDecision(
                    destination="cloud_anonymized",
                    detections=detections,
                    inspected_fields=inspected_fields,
                    reason="detected sensitive data, but operator prompt does not appear to need it",
                    query_text=query_text,
                    query_needs_sensitive=False,
                )
        else:
            decision = RouteDecision(
                destination="cloud",
                detections=[],
                inspected_fields=inspected_fields,
                reason="no sensitive data detected in operator inputs",
                query_text=query_text,
                query_needs_sensitive=False,
            )

        self.stats.record(decision)
        return decision

    def route(self, operator, input_fields: list[str], input_record: Any | None = None) -> str:
        """Compatibility wrapper returning only the destination string."""
        decision = self.inspect(operator, input_fields, input_record=input_record)
        self.last_decision = decision
        return decision.destination

    def _anonymize_text(self, text: str) -> str:
        if not text:
            return text

        analyzer = _get_shared_analyzer()
        anonymizer = _get_shared_anonymizer()
        if analyzer is not None and anonymizer is not None:
            try:
                results = analyzer.analyze(text=text, language="en")
                results = [
                    r for r in results
                    if getattr(r, "entity_type", "") in self.config.sensitive_entities
                    and (getattr(r, "score", 0.0) or 0.0) >= self.config.score_threshold
                ]
                if results:
                    return anonymizer.anonymize(text=text, analyzer_results=results).text
            except Exception:
                pass

        anonymized = text
        for entity_type, pattern in self._REGEX_PATTERNS.items():
            anonymized = pattern.sub(f"<{entity_type}>", anonymized)
        return anonymized

    def anonymize_record(self, input_record: Any, decision: RouteDecision) -> Any:
        try:
            anonymized_record = copy.copy(input_record)
        except Exception:
            anonymized_record = input_record

        fields_with_pii = {d.field_name for d in decision.detections}
        for field_name in fields_with_pii:
            try:
                value = getattr(input_record, field_name, "")
                text = self._coerce_text(value)
                new_value = self._anonymize_text(text)

                field_tokens = set(re.split(r"[^a-z0-9]+", field_name.lower())) - {""}
                if new_value == text and field_tokens & self._SENSITIVE_FIELD_HINTS:
                    new_value = f"<{field_name.upper()}_REDACTED>"

                setattr(anonymized_record, field_name, new_value)
            except Exception:
                continue
        return anonymized_record


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

    generated_fields = operator.generated_fields
    op_name = operator.op_name()
    model_name = operator.get_model_name()

    if cached_decision is not None:
        # Document-level: reuse a pre-scanned decision, but still tally stats
        # for this operator call so the benchmark counts are per-operator.
        decision = RouteDecision(
            destination=cached_decision.destination,
            detections=cached_decision.detections,
            inspected_fields=cached_decision.inspected_fields,
            reason=f"document-level cached: {cached_decision.reason}",
            query_text=cached_decision.query_text,
            query_needs_sensitive=cached_decision.query_needs_sensitive,
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
        f"  |  query_needs_sensitive={decision.query_needs_sensitive}"
        f"  |  query={router._safe_preview(decision.query_text, limit=100)!r}"
        f"  |  detections={detection_summary}"
    )

    record_for_execution = input_record
    if destination == "cloud_anonymized":
        record_for_execution = router.anonymize_record(input_record, decision)

    return operator(record_for_execution)

# ---------------------------------------------------------------------------
# Quick smoke-test (run this file directly to verify importability)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = ModelConfig()
    print(f"ModelConfig: local={config.local_model!r}  api_base={config.local_api_base!r}  cloud={config.cloud_model!r}")

    router = PrivacyRouter(config)

    class _FakeOperator:
        generated_fields = ["topic"]
        model = "gpt-4o"
        desc = "Determine whether the email discusses scheduling or travel."

        def op_name(self):
            return "LLMConvertBonded"

        def get_input_fields(self):
            return ["filename", "contents"]

        def get_model_name(self):
            return "openai/gpt-4o-2024-08-06"

        def __call__(self, record):
            assert "john@example.com" not in record.contents
            assert "617-555-1212" not in record.contents
            print(f"Operator executed with contents={record.contents!r}")
            return None

    class _SensitiveFakeOperator(_FakeOperator):
        generated_fields = ["sender"]
        desc = "Extract the sender email address from the email text."

    class _FakeRecord:
        filename = "email1.txt"
        contents = "Contact John Smith at john@example.com or 617-555-1212."

    fake_op = _FakeOperator()
    fake_record = _FakeRecord()

    decision = router.inspect(fake_op, fake_op.get_input_fields(), input_record=fake_record)
    print(f"Routing decision for non-sensitive query: {decision.destination!r}")
    assert decision.destination == "cloud_anonymized"

    sensitive_op = _SensitiveFakeOperator()
    local_decision = router.inspect(sensitive_op, sensitive_op.get_input_fields(), input_record=fake_record)
    print(f"Routing decision for sensitive query: {local_decision.destination!r}")
    assert local_decision.destination == "local"

    print(f"Stats after two inspections: {router.stats.summary()}")
    execute_with_routing(fake_op, fake_record, router)
    print("routing_stub.py: all checks passed.")
