"""
Privacy-aware routing stub for Palimpzest.

Provides:
  - ModelConfig       — local vs. cloud model identifiers
  - PrivacyRouter     — decides "local" or "cloud" per operator
  - execute_with_routing — wraps a single operator call with routing logic

This module is fully importable without runtime errors and requires no
modifications to PZ source files in src/palimpzest/.

Current behavior:
  - Tries to use Presidio if installed.
  - Falls back to lightweight regex / field-name heuristics if Presidio is not
    available.
  - Logs which fields and entity types triggered the decision.

Future work:
  - Add query-intent awareness ("does this operator actually need the sensitive
    field?") before routing sensitive-but-irrelevant inputs locally.
  - Swap `operator.model` for the local model in a PZ-native way once the team
    settles on the right model enum / config path.
"""

from __future__ import annotations

import sys
import os
import re

from dataclasses import dataclass, field
from typing import Any

# Ensure the PZ src is importable when this module is imported directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Maps the two routing destinations to actual model identifiers.

    local_model  — served locally via Ollama (no data leaves the machine)
    cloud_model  — served via OpenAI API  (highest quality, but data is sent externally)
    """
    local_model: str = "llama3.2"       # Ollama model tag
    cloud_model: str = "gpt-4o"         # OpenAI model identifier


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


# ---------------------------------------------------------------------------
# Privacy Router
# ---------------------------------------------------------------------------

class PrivacyRouter:
    """
    Decides whether a given operator should run against a local or cloud model.

    Routing policy (v1):
      - inspect the actual input values of the fields the operator reads
      - if any field appears to contain PII -> route to "local"
      - otherwise -> route to "cloud"

    The router prefers Presidio when available, but still works without it using
    field-name and regex heuristics. This makes the file importable even before
    the privacy extra is installed.
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
        self._analyzer = None
        self._presidio_error: Exception | None = None
        self.last_decision: RouteDecision | None = None

    def _get_analyzer(self):
        """Lazily construct Presidio's AnalyzerEngine if available."""
        if self._analyzer is not None:
            return self._analyzer
        if self._presidio_error is not None:
            return None

        try:
            from presidio_analyzer import AnalyzerEngine

            self._analyzer = AnalyzerEngine()
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._presidio_error = exc
            self._analyzer = None
        return self._analyzer

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
        """Run Presidio on a field value, returning normalized detections."""
        analyzer = self._get_analyzer()
        if analyzer is None or not text.strip():
            return []

        try:
            results = analyzer.analyze(text=text, language="en")
        except Exception:
            return []

        detections: list[Detection] = []
        for result in results:
            start = max(0, getattr(result, "start", 0))
            end = min(len(text), getattr(result, "end", 0))
            snippet = text[start:end] if end > start else text[:40]
            detections.append(
                Detection(
                    field_name=field_name,
                    entity_type=getattr(result, "entity_type", "PII"),
                    source="presidio",
                    score=getattr(result, "score", None),
                    preview=self._safe_preview(snippet),
                )
            )
        return detections

    def _detect_with_heuristics(self, field_name: str, text: str) -> list[Detection]:
        """Fallback detector using field-name hints and regexes."""
        detections: list[Detection] = []
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

        if not text.strip():
            return detections

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
            return RouteDecision(
                destination="local",
                detections=detections,
                inspected_fields=inspected_fields,
                reason="detected sensitive data in operator inputs",
            )

        return RouteDecision(
            destination="cloud",
            detections=[],
            inspected_fields=inspected_fields,
            reason="no sensitive data detected in operator inputs",
        )

    def route(self, operator, input_fields: list[str], input_record: Any | None = None) -> str:
        """Compatibility wrapper returning only the destination string."""
        decision = self.inspect(operator, input_fields, input_record=input_record)
        self.last_decision = decision
        return decision.destination


# ---------------------------------------------------------------------------
# Execution wrapper
# ---------------------------------------------------------------------------

def _set_operator_model_if_possible(operator, chosen_model: str) -> bool:
    """
    Best-effort model override.

    This intentionally stays conservative: if the operator doesn't expose a
    writable `model` attribute in a compatible format, we leave it untouched.
    """
    if not hasattr(operator, "model"):
        return False

    current_model = getattr(operator, "model")

    # Common/simple case: the operator stores a plain string.
    if isinstance(current_model, str):
        setattr(operator, "model", chosen_model)
        return True

    # Anything more complicated is PZ-internal; skip rather than risk breaking.
    return False



def execute_with_routing(operator, input_record, router: PrivacyRouter):
    """
    Wrap a single operator execution with privacy-aware model routing.

    This function:
      1. Inspects the operator's input fields and actual input values.
      2. Calls router.route() to determine the execution venue.
      3. Logs the routing decision, trigger fields, and entity types.
      4. Best-effort swaps operator.model when feasible.
      5. Executes the operator via operator(input_record).

    Args:
        operator:      A PZ PhysicalOperator instance.
        input_record:  A pz DataRecord to pass to the operator.
        router:        A PrivacyRouter instance.

    Returns:
        DataRecordSet — the output from operator(input_record).
    """
    # Gather field context for routing decision
    input_fields: list[str] = operator.get_input_fields()
    generated_fields: list[str] = operator.generated_fields
    op_name: str = operator.op_name()
    model_name = operator.get_model_name()
    # Ask the router where this operator should run
    decision = router.inspect(operator, input_fields, input_record=input_record)
    router.last_decision = decision
    destination = decision.destination

    # Log the decision
    chosen_model = (
        router.config.cloud_model if destination == "cloud"
        else router.config.local_model
    )
    model_swapped = _set_operator_model_if_possible(operator, chosen_model)

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

    # Execute through PZ's standard path (no model swap yet)
    record_set = operator(input_record)

    return record_set


# ---------------------------------------------------------------------------
# Quick smoke-test (run this file directly to verify importability)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Verify the module is importable and the classes instantiate correctly
    config = ModelConfig()
    print(f"ModelConfig: local={config.local_model!r}  cloud={config.cloud_model!r}")

    router = PrivacyRouter(config)

    # Simulate a call with a dummy operator-like object
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
    execute_with_routing(fake_op, fake_record, router)
    print("routing_stub.py: all checks passed.")
