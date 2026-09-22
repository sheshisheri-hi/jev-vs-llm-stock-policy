"""Core policy, schema, gate, parser, and mocked Jev HTTP tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import httpx2
import pytest

from stock_policy.compare import render_table, run_all
from stock_policy.fixtures import FIXTURES, get_fixture, portfolio_sums_match
from stock_policy.gate import (
    FRAUD_DENY,
    MIN_CHOICE_CONFIDENCE,
    JevReadout,
    action_from_jev,
    apply_gate,
)
from stock_policy.jev_client import (
    JEV_MODEL,
    decide as decide_jev,
    policy_questions,
    resolve_api_key,
)
from stock_policy.llm_client import parse_llm_text, stub_text
from stock_policy.policy import VALID_ACTIONS, evaluate_hard_rules

ROOT = Path(__file__).resolve().parents[1]


def _quiet_allow(**overrides: object) -> JevReadout:
    base = dict(
        source="test",
        model="test",
        action="allow",
        action_confidence=0.93,
        action_probabilities={"allow": 0.93, "escalate_to_human": 0.05, "deny": 0.02},
        risk_score=0.1,
        risk_confidence=0.9,
        risk_label="Low",
        risk_probabilities={"Low": 0.9, "Medium": 0.08, "High": 0.02, "Critical": 0.0},
        outside_policy=0.05,
        looks_like_exfil_or_fraud=0.02,
        note="unit",
    )
    base.update(overrides)
    return JevReadout(**base)  # type: ignore[arg-type]


def test_fixture_loader_has_six_synthetic_tickets() -> None:
    assert len(FIXTURES) == 6
    ids = [fixture.id for fixture in FIXTURES]
    assert ids == [
        "core-top-up",
        "routine-hold",
        "concentration-breach",
        "after-hours-options",
        "sell-all-offshore",
        "gray-zone-tip",
    ]
    assert len(set(ids)) == 6
    for fixture in FIXTURES:
        assert portfolio_sums_match(fixture)
        assert fixture.expected_final in VALID_ACTIONS
        assert "http" not in fixture.rationale.lower()
    assert get_fixture("gray-zone-tip").title.startswith("Gray-zone")
    with pytest.raises(KeyError):
        get_fixture("not-a-ticket")


def test_choice_schema_is_only_the_three_policy_actions() -> None:
    questions = policy_questions()
    assert set(questions) == {"action", "risk", "outside_policy", "looks_like_exfil_or_fraud"}
    action = questions["action"]
    assert action.type == "choice"
    assert set(action.criteria) == set(VALID_ACTIONS)
    assert questions["risk"].type == "score"
    assert [str(level)[:3] for level in questions["risk"].criteria] == ["Low", "Med", "Hig", "Cri"]
    assert questions["outside_policy"].type == "noul"
    assert questions["looks_like_exfil_or_fraud"].type == "noul"


def test_hard_rules_deny_blocklist_and_notional() -> None:
    offshore = get_fixture("sell-all-offshore")
    denied = evaluate_hard_rules(offshore)
    assert denied.action == "deny"
    assert any("blocklist" in reason for reason in denied.reasons)
    assert any("notional" in reason for reason in denied.reasons)

    small_authorized = replace(
        offshore,
        order=replace(
            offshore.order,
            notional_usd=1_000,
            intents=("sell_all", "wire_out", "offshore_transfer"),
            multi_party_auth=True,
            symbol="CASH",
        ),
    )
    assert evaluate_hard_rules(small_authorized).action is None


def test_hard_rules_escalate_concentration_delta_and_after_hours() -> None:
    assert evaluate_hard_rules(get_fixture("concentration-breach")).action == "escalate_to_human"
    options = evaluate_hard_rules(get_fixture("after-hours-options"))
    assert options.action == "escalate_to_human"
    assert any("delta" in reason for reason in options.reasons)
    assert any("after-hours" in reason for reason in options.reasons)
    assert evaluate_hard_rules(get_fixture("core-top-up")).action is None
    assert evaluate_hard_rules(get_fixture("routine-hold")).action is None
    assert evaluate_hard_rules(get_fixture("gray-zone-tip")).action is None

    options_ticket = get_fixture("after-hours-options")
    at_threshold = replace(
        options_ticket,
        order=replace(options_ticket.order, delta=0.40, notional_usd=1_000, session="regular"),
    )
    assert evaluate_hard_rules(at_threshold).action is None
    just_over = replace(at_threshold, order=replace(at_threshold.order, delta=0.41))
    assert evaluate_hard_rules(just_over).action == "escalate_to_human"

    tiny_after_hours = replace(
        options_ticket,
        order=replace(
            options_ticket.order,
            asset_class="equity",
            delta=None,
            intents=("equity_buy",),
            notional_usd=24_999,
            session="after_hours",
            symbol="TINY",
        ),
    )
    assert evaluate_hard_rules(tiny_after_hours).action is None
    aggressive = replace(tiny_after_hours, order=replace(tiny_after_hours.order, notional_usd=25_000))
    assert evaluate_hard_rules(aggressive).action == "escalate_to_human"


def test_gate_hard_deny_beats_model_allow() -> None:
    hard = evaluate_hard_rules(get_fixture("sell-all-offshore"))
    decision = apply_gate(hard, _quiet_allow())
    assert decision.action == "deny"
    assert decision.model_action == "allow"
    assert "beats model allow" in decision.reason


def test_gate_hard_escalate_beats_model_allow_and_model_deny_is_stricter() -> None:
    hard = evaluate_hard_rules(get_fixture("concentration-breach"))
    held = apply_gate(hard, _quiet_allow())
    assert held.action == "escalate_to_human"
    assert "beats model allow" in held.reason

    denial = apply_gate(
        hard,
        _quiet_allow(
            action="deny",
            action_confidence=0.91,
            action_probabilities={"allow": 0.02, "escalate_to_human": 0.07, "deny": 0.91},
            looks_like_exfil_or_fraud=FRAUD_DENY,
        ),
    )
    assert denial.action == "deny"


def test_gate_thresholds_on_typed_fields() -> None:
    no_rule_ticket = get_fixture("core-top-up")
    assert evaluate_hard_rules(no_rule_ticket).action is None

    low_confidence = apply_gate(
        evaluate_hard_rules(no_rule_ticket),
        _quiet_allow(action_confidence=MIN_CHOICE_CONFIDENCE - 0.01),
    )
    assert low_confidence.action == "escalate_to_human"
    assert "confidence" in low_confidence.reason

    critical = apply_gate(
        evaluate_hard_rules(no_rule_ticket),
        _quiet_allow(risk_label="Critical", risk_score=3.0),
    )
    assert critical.action == "escalate_to_human"

    fraud = apply_gate(
        evaluate_hard_rules(no_rule_ticket),
        _quiet_allow(looks_like_exfil_or_fraud=0.91, action="allow"),
    )
    assert fraud.action == "deny"

    invented = action_from_jev(_quiet_allow(action="maybe_later", action_confidence=0.99))
    assert invented[0] == "escalate_to_human"


def test_parser_rejects_invented_llm_actions() -> None:
    assert parse_llm_text('{"action": "maybe_later"}').parsed_action is None
    assert parse_llm_text('{"action": "maybe_later"}').invented_action == "maybe_later"
    assert parse_llm_text("approve_with_vibes and ship it").parsed_action is None
    assert parse_llm_text('{"decision": "wire_it"}').parsed_action is None
    assert parse_llm_text('{"action": "deny"}').parsed_action == "deny"
    assert parse_llm_text("I would allow this small add.").parsed_action == "allow"
    mixed = parse_llm_text("maybe allow, but really approve_with_vibes")
    assert mixed.parsed_action is None
    assert mixed.clamped_action == "escalate_to_human"

    assert parse_llm_text(stub_text("core-top-up")).parsed_action == "allow"
    for fixture_id in (
        "routine-hold",
        "concentration-breach",
        "after-hours-options",
        "sell-all-offshore",
        "gray-zone-tip",
    ):
        parsed = parse_llm_text(stub_text(fixture_id))
        assert parsed.parsed_action is None
        assert parsed.invented_action in {"maybe_later", "approve_with_vibes", "wire_it", "yolo_buy"}


def test_sample_path_matches_expected_finals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    comparisons = run_all(live_jev=False, live_llm=False)
    assert [item.fixture_id for item in comparisons] == [fixture.id for fixture in FIXTURES]
    for comparison in comparisons:
        assert comparison.jev.source == "sample"
        assert comparison.llm.source == "stub"
        assert comparison.gate.action == comparison.expected_final
        assert comparison.jev.action in VALID_ACTIONS
        assert set(comparison.jev.action_probabilities) == set(VALID_ACTIONS)
    table = render_table(comparisons)
    for header in ("Input", "LLM", "Jev", "Final gate"):
        assert header in table
    assert "OUT OF SCHEMA" in table
    assert "SAMPLE" in table


def test_api_key_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    assert resolve_api_key() is None

    monkeypatch.setenv("JEV_API_KEY", "alias-key")
    assert resolve_api_key() == "alias-key"
    assert os.environ["TYPESAFE_API_KEY"] == "alias-key"

    monkeypatch.setenv("TYPESAFE_API_KEY", "primary-key")
    monkeypatch.setenv("JEV_API_KEY", "other")
    assert resolve_api_key() == "primary-key"


class _CaptureTransport(httpx2.BaseTransport):
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.request: httpx2.Request | None = None

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        self.request = request
        return httpx2.Response(
            self.status,
            headers={"content-type": "application/json"},
            content=json.dumps(self.payload).encode(),
        )


def _live_payload(*, choice: str = "deny") -> dict[str, object]:
    return {
        "model": "jev-1.13.0",
        "usage": {"input_tokens": 120, "output_tokens": 18},
        "answers": {
            "action": {
                "type": "choice",
                "choice": choice,
                "confidence": 0.91,
                "probabilities": {"allow": 0.04, "escalate_to_human": 0.05, "deny": 0.91},
            },
            "risk": {
                "type": "score",
                "score": 2.8,
                "confidence": 0.88,
                "legend": {"0": "Low", "1": "Medium", "2": "High", "3": "Critical"},
                "probabilities": {"0": 0.01, "1": 0.04, "2": 0.15, "3": 0.80},
            },
            "outside_policy": {"type": "noul", "noul": 0.93},
            "looks_like_exfil_or_fraud": {"type": "noul", "noul": 0.97},
        },
    }


def test_jev_client_posts_system_one_and_reads_probabilities() -> None:
    transport = _CaptureTransport(_live_payload())
    readout = decide_jev(
        get_fixture("sell-all-offshore"),
        live=True,
        api_key="test-key",
        transport=transport,
    )
    assert transport.request is not None
    assert transport.request.method == "POST"
    assert transport.request.url.path == "/v1/systemone"
    assert transport.request.headers["Authorization"] == "Bearer test-key"
    body = json.loads(transport.request.content.decode())
    assert body["model"] == JEV_MODEL
    assert set(body["questions"]) == {"action", "risk", "outside_policy", "looks_like_exfil_or_fraud"}
    assert body["questions"]["action"]["type"] == "choice"
    assert set(body["questions"]["action"]["criteria"]) == set(VALID_ACTIONS)
    assert body["questions"]["risk"]["type"] == "score"
    assert body["questions"]["outside_policy"]["type"] == "noul"
    assert body["questions"]["looks_like_exfil_or_fraud"]["type"] == "noul"
    assert "agent_rationale" in body["state"]
    assert readout.source == "live"
    assert readout.model == "jev-1.13.0"
    assert readout.action == "deny"
    assert readout.action_probabilities["deny"] == pytest.approx(0.91)
    assert readout.risk_label == "Critical"
    assert readout.outside_policy == pytest.approx(0.93)
    assert readout.looks_like_exfil_or_fraud == pytest.approx(0.97)


def test_jev_client_rejects_out_of_schema_choice() -> None:
    transport = _CaptureTransport(_live_payload(choice="maybe_later"))
    with pytest.raises(ValueError, match="outside"):
        decide_jev(
            get_fixture("core-top-up"),
            live=True,
            api_key="test-key",
            transport=transport,
        )


def test_demo_cli_prints_six_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env = os.environ.copy()
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "demo.py")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    for header in ("Input", "LLM", "Jev", "Final gate"):
        assert header in completed.stdout
    for fixture in FIXTURES:
        assert fixture.id in completed.stdout
    assert "SAMPLE" in completed.stdout
    assert "OUT OF SCHEMA" in completed.stdout

    listed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "demo.py"), "--json", "--fixture", "core-top-up"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert listed.returncode == 0, listed.stderr
    payload = json.loads(listed.stdout)
    assert len(payload) == 1
    assert payload[0]["final_gate"]["action"] == "allow"

    missing = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "demo.py"), "--live-jev"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 2
    assert "TYPESAFE_API_KEY" in missing.stderr

    missing_llm = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "demo.py"), "--live-llm"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing_llm.returncode == 2
    assert "OPENAI_API_KEY" in missing_llm.stderr

    unknown = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "demo.py"), "--fixture", "nope"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert unknown.returncode == 2
    assert "core-top-up" in unknown.stderr
