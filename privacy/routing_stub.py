"""
Privacy-aware routing stub for Palimpzest.

Provides:
  - ModelConfig       — local vs. cloud model identifiers
  - PrivacyRouter     — decides "local" or "cloud" per operator (currently hardcoded)
  - execute_with_routing — wraps a single operator call with routing logic

This module is fully importable without runtime errors and requires no
modifications to PZ source files in src/palimpzest/.

Future work:
  - Replace the hardcoded "always cloud" rule with Presidio-based PII detection
    (see OPERATOR_GRAPH_NOTES.md §6 for the integration plan).
  - Swap `operator.model` for the local model when PrivacyRouter returns "local".
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

# Ensure the PZ src is importable when this module is imported directly
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


# ---------------------------------------------------------------------------
# Privacy Router
# ---------------------------------------------------------------------------

class PrivacyRouter:
    """
    Decides whether a given operator should run against a local or cloud model.

    The routing decision is based on:
      - The operator type (e.g. LLMConvertBonded, LLMFilter)
      - The list of input field names that the operator will read

    Current implementation: always returns "cloud" (placeholder).

    Future implementation (Presidio-based):
        from presidio_analyzer import AnalyzerEngine
        engine = AnalyzerEngine()
        for field_name in input_fields:
            field_value = getattr(input_record, field_name, "") or ""
            results = engine.analyze(text=field_value, language="en")
            if results:
                return "local"
        return "cloud"
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()

    def route(self, operator, input_fields: list[str]) -> str:
        """
        Decide routing for a single operator invocation.

        Args:
            operator:     A PZ PhysicalOperator instance (has .op_name(),
                          .input_schema, .output_schema, .generated_fields,
                          .get_model_name(), .get_input_fields()).
            input_fields: List of field name strings the operator will read
                          (from operator.get_input_fields() or custom logic).

        Returns:
            "local"  — use the local Ollama model (data stays on-device)
            "cloud"  — use the cloud OpenAI model  (data sent to external API)
        """
        # TODO: Replace with Presidio-based PII detection.
        # Example skeleton:
        #
        #   from presidio_analyzer import AnalyzerEngine
        #   _PII_FIELDS = {"email", "name", "phone", "ssn", "address",
        #                  "credit_card", "ip_address", "person"}
        #   for fname in input_fields:
        #       if fname.lower() in _PII_FIELDS:
        #           return "local"
        #   return "cloud"

        return "cloud"   # hardcoded placeholder


# ---------------------------------------------------------------------------
# Execution wrapper
# ---------------------------------------------------------------------------

def execute_with_routing(operator, input_record, router: PrivacyRouter):
    """
    Wrap a single operator execution with privacy-aware model routing.

    This function:
      1. Inspects the operator's input fields.
      2. Calls router.route() to determine the execution venue.
      3. Logs the routing decision (operator name, fields, chosen destination).
      4. Executes the operator normally via operator(input_record), which is
         PZ's standard invocation path (PhysicalOperator.__call__).

    NOTE: This function does NOT yet swap operator.model.  In a real deployment
    you would set operator.model = Model(router.config.local_model) when the
    decision is "local".  That requires the model string to be a valid PZ
    Model enum value (see src/palimpzest/constants.py).

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
    destination = router.route(operator, input_fields)

    # Log the decision
    chosen_model = (
        router.config.cloud_model if destination == "cloud"
        else router.config.local_model
    )
    print(
        f"[PrivacyRouter] {op_name}"
        f"  |  input_fields={input_fields}"
        f"  |  generates={generated_fields}"
        f"  |  current_model={model_name}"
        f"  |  routed_to={destination!r}  ({chosen_model})"
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
        def op_name(self): return "LLMConvertBonded"
        def get_input_fields(self): return ["filename", "contents"]
        def get_model_name(self): return "openai/gpt-4o-2024-08-06"
        generated_fields = ["subject", "sender"]

        def __call__(self, record):
            return None  # placeholder

    fake_op = _FakeOperator()
    decision = router.route(fake_op, fake_op.get_input_fields())
    print(f"Routing decision for {fake_op.op_name()!r}: {decision!r}")

    assert decision == "cloud", "Expected 'cloud' for hardcoded placeholder"
    print("routing_stub.py: all checks passed.")
