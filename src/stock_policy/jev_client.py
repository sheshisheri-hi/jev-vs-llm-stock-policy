"""TypeSafe Jev (System One) client.

One POST per order to https://api.typesafe.ai/v1/systemone via typesafe-sdk.
Parallel questions: Choice action, Score risk, two Nouls. The response is a
distribution over the schema you declared. It is not free-form text.

Without an API key, `decide` returns an authored SAMPLE readout so the demo
still runs. SAMPLE rows are schema-shaped illustrations, not live model traces.
"""

from __future__ import annotations

import os
from typing import Any

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

from stock_policy.fixtures import Fixture
from stock_policy.gate import JevReadout, risk_label_from_probabilities
from stock_policy.policy import VALID_ACTIONS, ticket_state

JEV_MODEL = "jev-latest"
API_KEY_ENV = "TYPESAFE_API_KEY"
API_KEY_ALIAS_ENV = "JEV_API_KEY"


class MissingJevKeyError(RuntimeError):
    """Raised when a live call was requested and neither key is set."""


def resolve_api_key(explicit: str | None = None) -> str | None:
    """Prefer TYPESAFE_API_KEY. Accept JEV_API_KEY and copy it onto the SDK env var."""

    if explicit is not None and explicit.strip():
        return explicit.strip()
    primary = os.environ.get(API_KEY_ENV, "").strip()
    if primary:
        return primary
    alias = os.environ.get(API_KEY_ALIAS_ENV, "").strip()
    if not alias:
        return None
    os.environ[API_KEY_ENV] = alias
    return alias


def policy_questions() -> dict[str, Choice | Noul | Score]:
    """The four questions asked together on every ticket."""

    return {
        "action": Choice(
            instructions=(
                "Choose the order-policy action. "
                "allow only when the ticket fits the written policy and the rationale "
                "does not describe fraud, exfiltration, or material non-public information. "
                "escalate_to_human when a person should review mixed or uncomfortable signals. "
                "deny when the ticket should be blocked."
            ),
            criteria={
                "allow": "Release the order. It fits the written policy.",
                "escalate_to_human": "Do not release it. A person must review.",
                "deny": "Block the order. It is a prohibited transfer or a clear breach.",
            },
        ),
        "risk": Score(
            instructions="Rate the policy risk of releasing this order as proposed.",
            criteria=[
                "Low: routine ticket inside the written limits, no fraud or tip language",
                "Medium: notable, but still a normal desk judgment",
                "High: material policy concern, ambiguous intent, or uncomfortable rationale",
                "Critical: likely fraud, exfiltration, or a clear request to bypass controls",
            ],
        ),
        "outside_policy": Noul(
            instructions=(
                "Does this order violate the stated desk policy "
                "(size, concentration, options delta, after-hours size, or the blocklist)?"
            ),
            criteria={
                "true": "The order breaks a written limit or a blocklist rule.",
                "false": "The order sits inside the written policy.",
            },
        ),
        "looks_like_exfil_or_fraud": Noul(
            instructions=(
                "Does this request look like account exfiltration or fraud "
                "(sell everything, wire out, offshore transfer, or a brand-new destination)?"
            ),
            criteria={
                "true": "The pattern matches a dump, a wire-out, or an offshore transfer.",
                "false": "The request is an ordinary portfolio instruction.",
            },
        ),
    }


def _readout(
    *,
    source: str,
    model: str,
    action: str,
    action_confidence: float,
    action_probabilities: dict[str, float],
    risk_score: float,
    risk_confidence: float,
    risk_probabilities: dict[str, float],
    outside_policy: float,
    looks_like_exfil_or_fraud: float,
    note: str,
) -> JevReadout:
    by_index = {index: risk_probabilities[label] for index, label in enumerate(("Low", "Medium", "High", "Critical"))}
    return JevReadout(
        source=source,
        model=model,
        action=action,
        action_confidence=action_confidence,
        action_probabilities=dict(action_probabilities),
        risk_score=risk_score,
        risk_confidence=risk_confidence,
        risk_label=risk_label_from_probabilities(by_index),
        risk_probabilities=dict(risk_probabilities),
        outside_policy=outside_policy,
        looks_like_exfil_or_fraud=looks_like_exfil_or_fraud,
        note=note,
    )


# Authored distributions. They exist so `demo.py` runs with no key.
# They are not measurements of jev-latest.
_SAMPLE_NOTE = "authored SAMPLE, schema-valid, not a live jev-latest call"

SAMPLE_BY_ID: dict[str, JevReadout] = {
    "core-top-up": _readout(
        source="sample",
        model="sample",
        action="allow",
        action_confidence=0.92,
        action_probabilities={"allow": 0.94, "escalate_to_human": 0.05, "deny": 0.01},
        risk_score=0.12,
        risk_confidence=0.90,
        risk_probabilities={"Low": 0.90, "Medium": 0.08, "High": 0.02, "Critical": 0.00},
        outside_policy=0.04,
        looks_like_exfil_or_fraud=0.01,
        note=_SAMPLE_NOTE,
    ),
    "routine-hold": _readout(
        source="sample",
        model="sample",
        action="allow",
        action_confidence=0.90,
        action_probabilities={"allow": 0.92, "escalate_to_human": 0.07, "deny": 0.01},
        risk_score=0.10,
        risk_confidence=0.91,
        risk_probabilities={"Low": 0.92, "Medium": 0.06, "High": 0.02, "Critical": 0.00},
        outside_policy=0.05,
        looks_like_exfil_or_fraud=0.01,
        note=_SAMPLE_NOTE,
    ),
    "concentration-breach": _readout(
        source="sample",
        model="sample",
        # Authored as allow on purpose, so the demo shows the hard rule beating a model allow.
        # This is not a recording of jev-latest.
        action="allow",
        action_confidence=0.88,
        action_probabilities={"allow": 0.88, "escalate_to_human": 0.09, "deny": 0.03},
        risk_score=1.05,
        risk_confidence=0.74,
        risk_probabilities={"Low": 0.16, "Medium": 0.68, "High": 0.13, "Critical": 0.03},
        outside_policy=0.22,
        looks_like_exfil_or_fraud=0.03,
        note=_SAMPLE_NOTE + "; Choice is allow so the concentration rule has an allow to override",
    ),
    "after-hours-options": _readout(
        source="sample",
        model="sample",
        action="escalate_to_human",
        action_confidence=0.86,
        action_probabilities={"allow": 0.09, "escalate_to_human": 0.86, "deny": 0.05},
        risk_score=1.70,
        risk_confidence=0.74,
        risk_probabilities={"Low": 0.04, "Medium": 0.28, "High": 0.60, "Critical": 0.08},
        outside_policy=0.36,
        looks_like_exfil_or_fraud=0.03,
        note=_SAMPLE_NOTE,
    ),
    "sell-all-offshore": _readout(
        source="sample",
        model="sample",
        action="deny",
        action_confidence=0.95,
        action_probabilities={"allow": 0.01, "escalate_to_human": 0.06, "deny": 0.93},
        risk_score=2.92,
        risk_confidence=0.93,
        risk_probabilities={"Low": 0.00, "Medium": 0.02, "High": 0.08, "Critical": 0.90},
        outside_policy=0.97,
        looks_like_exfil_or_fraud=0.98,
        note=_SAMPLE_NOTE,
    ),
    "gray-zone-tip": _readout(
        source="sample",
        model="sample",
        action="escalate_to_human",
        action_confidence=0.74,
        action_probabilities={"allow": 0.16, "escalate_to_human": 0.74, "deny": 0.10},
        risk_score=2.15,
        risk_confidence=0.71,
        risk_probabilities={"Low": 0.03, "Medium": 0.12, "High": 0.70, "Critical": 0.15},
        outside_policy=0.63,
        looks_like_exfil_or_fraud=0.28,
        note=_SAMPLE_NOTE,
    ),
}


def sample_readout(fixture_id: str) -> JevReadout:
    try:
        return SAMPLE_BY_ID[fixture_id]
    except KeyError as exc:
        known = ", ".join(SAMPLE_BY_ID)
        raise KeyError(f"No SAMPLE readout for {fixture_id!r}. Known ids: {known}") from exc


def _require_schema_action(action: str) -> str:
    if action not in VALID_ACTIONS:
        raise ValueError(
            f"Jev choice {action!r} is outside allow|escalate_to_human|deny. "
            "Refusing to treat it as a decision."
        )
    return action


def readout_from_response(response: Any, *, source: str) -> JevReadout:
    """Adapt a SystemOneResponse into the gate's readout. Validates the Choice label."""

    action_answer = response.choices["action"]
    risk_answer = response.scores["risk"]
    action = _require_schema_action(action_answer.choice)
    risk_probs_by_index = {int(level): float(prob) for level, prob in risk_answer.probabilities.items()}
    risk_labels = {
        ("Low", "Medium", "High", "Critical")[index]: risk_probs_by_index.get(index, 0.0)
        for index in range(4)
    }
    return JevReadout(
        source=source,
        model=str(response.model),
        action=action,
        action_confidence=float(action_answer.confidence),
        action_probabilities={str(name): float(prob) for name, prob in action_answer.probabilities.items()},
        risk_score=float(risk_answer.score),
        risk_confidence=float(risk_answer.confidence),
        risk_label=risk_label_from_probabilities(risk_probs_by_index),
        risk_probabilities=risk_labels,
        outside_policy=float(response.nouls["outside_policy"].noul),
        looks_like_exfil_or_fraud=float(response.nouls["looks_like_exfil_or_fraud"].noul),
        note="live System One response",
    )


def call_live(fixture: Fixture, *, api_key: str, transport: Any = None) -> JevReadout:
    """POST one System One request. `transport` is for tests that mock HTTP."""

    with TypeSafeClient(api_key=api_key, model=JEV_MODEL, transport=transport) as client:
        response = client.system_one(
            state=ticket_state(fixture),
            questions=policy_questions(),
            model=JEV_MODEL,
        )
    return readout_from_response(response, source="live")


def decide(
    fixture: Fixture,
    *,
    live: bool | None = None,
    api_key: str | None = None,
    transport: Any = None,
) -> JevReadout:
    """Live when a key is present, unless `live` is set. No key and live=False yields SAMPLE."""

    key = resolve_api_key(api_key)
    use_live = (key is not None) if live is None else live
    if not use_live:
        return sample_readout(fixture.id)
    if not key:
        raise MissingJevKeyError(
            "Live Jev was requested, but neither TYPESAFE_API_KEY nor JEV_API_KEY is set. "
            "Copy .env.example to .env and add a key, or run without --live-jev to use SAMPLE answers."
        )
    return call_live(fixture, api_key=key, transport=transport)
