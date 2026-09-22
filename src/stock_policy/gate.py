"""Combine hard rules with a typed Jev readout. The LLM is not an input.

Priority:
1. A hard deny always wins, including over a model allow.
2. A hard escalate is a floor. The model may still deny (stricter), but it
   cannot lower the floor to allow.
3. Otherwise the typed Jev fields decide: fraud / outside-policy / Critical
   risk, then Choice confidence, then the Choice label itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from stock_policy.policy import VALID_ACTIONS, HardRuleResult

RISK_LABELS = ("Low", "Medium", "High", "Critical")

# Thresholds are desk policy, owned by this repo. They are not properties of the model.
OUTSIDE_POLICY_ESCALATE = 0.55
OUTSIDE_POLICY_DENY = 0.85
FRAUD_ESCALATE = 0.50
FRAUD_DENY = 0.80
MIN_CHOICE_CONFIDENCE = 0.70


@dataclass(frozen=True)
class JevReadout:
    source: str
    model: str
    action: str
    action_confidence: float
    action_probabilities: dict[str, float]
    risk_score: float
    risk_confidence: float
    risk_label: str
    risk_probabilities: dict[str, float]
    outside_policy: float
    looks_like_exfil_or_fraud: float
    note: str = ""


@dataclass(frozen=True)
class GateDecision:
    action: str
    reason: str
    hard_action: str | None
    model_action: str
    hard_reasons: tuple[str, ...]


def risk_label_from_probabilities(probabilities: dict[int, float]) -> str:
    """Pick the rubric level with the highest probability. Ties break toward higher risk."""

    best = max(range(len(RISK_LABELS)), key=lambda index: (probabilities.get(index, 0.0), index))
    return RISK_LABELS[best]


def action_from_jev(jev: JevReadout) -> tuple[str, str]:
    """Map typed fields to one schema action. Never returns an out-of-schema label."""

    fraud = jev.looks_like_exfil_or_fraud
    outside = jev.outside_policy
    if fraud >= FRAUD_DENY:
        return "deny", f"fraud noul {fraud:.2f} >= {FRAUD_DENY:.2f}"
    if jev.risk_label == "Critical" and outside >= OUTSIDE_POLICY_DENY:
        return (
            "deny",
            f"risk Critical and outside_policy {outside:.2f} >= {OUTSIDE_POLICY_DENY:.2f}",
        )
    if jev.risk_label == "Critical":
        return "escalate_to_human", "risk Critical"
    if outside >= OUTSIDE_POLICY_ESCALATE:
        return (
            "escalate_to_human",
            f"outside_policy {outside:.2f} >= {OUTSIDE_POLICY_ESCALATE:.2f}",
        )
    if fraud >= FRAUD_ESCALATE:
        return (
            "escalate_to_human",
            f"fraud noul {fraud:.2f} >= {FRAUD_ESCALATE:.2f}",
        )
    if jev.action not in VALID_ACTIONS:
        return "escalate_to_human", f"choice {jev.action!r} is outside the schema"
    if jev.action_confidence < MIN_CHOICE_CONFIDENCE:
        return (
            "escalate_to_human",
            f"choice confidence {jev.action_confidence:.2f} < {MIN_CHOICE_CONFIDENCE:.2f}",
        )
    return jev.action, f"choice {jev.action} at confidence {jev.action_confidence:.2f}"


def apply_gate(hard: HardRuleResult, jev: JevReadout) -> GateDecision:
    model_action, model_reason = action_from_jev(jev)
    if model_action not in VALID_ACTIONS:
        model_action = "escalate_to_human"
        model_reason = "model action fell outside the schema; fail closed"

    if hard.action == "deny":
        reason = f"hard deny beats model {model_action} ({model_reason})"
        return GateDecision("deny", reason, hard.action, model_action, hard.reasons)

    if hard.action == "escalate_to_human":
        if model_action == "deny":
            final = "deny"
            reason = f"model deny ({model_reason}) is stricter than the hard escalate floor"
        elif model_action == "allow":
            final = "escalate_to_human"
            reason = f"hard escalate beats model allow ({model_reason})"
        else:
            final = "escalate_to_human"
            reason = f"hard escalate floor; model also {model_action} ({model_reason})"
        return GateDecision(final, reason, hard.action, model_action, hard.reasons)

    return GateDecision(model_action, model_reason, None, model_action, ())
