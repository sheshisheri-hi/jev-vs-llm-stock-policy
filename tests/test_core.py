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
from stock_policy.llm_client import (
    CURSOR_HTTP_TIMEOUT_SECONDS,
    CURSOR_RUN_TIMEOUT_SECONDS,
    decide as decide_llm,
    decide_many,
    live_llm_ready,
    parse_llm_text,
    prompt_for,
    resolve_llm_credentials,
    stub_text,
)
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
    for name in (
        "TYPESAFE_API_KEY",
        "JEV_API_KEY",
        "OPENAI_API_KEY",
        "GROK_BOT_API_KEY",
        "CURSOR_API_KEY",
    ):
        monkeypatch.setenv(name, "")
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
    assert "CURSOR_API_KEY" in missing_llm.stderr

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


class _JsonResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._raw = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_JsonResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _install_urlopen(monkeypatch: pytest.MonkeyPatch, handler):  # type: ignore[no-untyped-def]
    monkeypatch.setattr("stock_policy.llm_client.time.sleep", lambda _delay: None)
    monkeypatch.setattr("stock_policy.llm_client.urllib.request.urlopen", handler)


def test_llm_key_resolution_order(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "GROK_BOT_API_KEY", "CURSOR_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert resolve_llm_credentials() == (None, "missing")
    assert live_llm_ready() is False

    monkeypatch.setenv("CURSOR_API_KEY", "crsr_cursor")
    monkeypatch.setenv("GROK_BOT_API_KEY", "desk-bot-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert resolve_llm_credentials() == ("sk-openai", "openai")

    monkeypatch.setenv("OPENAI_API_KEY", "crsr_in_openai_slot")
    assert resolve_llm_credentials() == ("desk-bot-key", "cursor")

    monkeypatch.delenv("GROK_BOT_API_KEY")
    assert resolve_llm_credentials() == ("crsr_cursor", "cursor")

    monkeypatch.delenv("CURSOR_API_KEY")
    assert resolve_llm_credentials() == ("crsr_in_openai_slot", "cursor")

    assert resolve_llm_credentials("crsr_explicit") == ("crsr_explicit", "cursor")
    assert resolve_llm_credentials("sk-explicit") == ("sk-explicit", "openai")


def test_stub_ignores_cursor_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "crsr_test")

    def fail_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("live LLM should not be called when live=False")

    _install_urlopen(monkeypatch, fail_open)
    result = decide_llm(get_fixture("core-top-up"), live=False)
    assert result.source == "stub"
    rows = run_all(live_jev=False, live_llm=False, fixture_id="core-top-up")
    assert rows[0].llm.source == "stub"


def test_cursor_key_reuses_one_agent_then_archives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_BOT_API_KEY", raising=False)
    monkeypatch.setenv("CURSOR_API_KEY", "crsr_test")
    calls: list[tuple[str, str, dict[str, object] | None]] = []
    second_polls = {"n": 0}

    def fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        import base64

        assert request.get_header("Authorization") == "Basic " + base64.b64encode(b"crsr_test:").decode()
        method = request.get_method()
        url = request.full_url
        body = json.loads(request.data.decode()) if request.data else None
        calls.append((method, url, body))
        if method == "POST" and url == "https://api.cursor.com/v1/agents":
            assert body is not None
            assert body["name"] == "jev-demo-llm"
            assert "repos" not in body
            assert "env" not in body
            assert body["prompt"]["text"] == prompt_for(get_fixture("core-top-up"))
            return _JsonResponse({"agent": {"id": "ag_1"}, "run": {"id": "run_1"}})
        if method == "GET" and url.endswith("/runs/run_1"):
            return _JsonResponse({"status": "FINISHED", "result": "Small add, inside the sleeve.\nallow"})
        if method == "POST" and url == "https://api.cursor.com/v1/agents/ag_1/runs":
            assert body is not None
            assert body["prompt"]["text"] == prompt_for(get_fixture("sell-all-offshore"))
            return _JsonResponse({"run": {"id": "run_2"}})
        if method == "GET" and url.endswith("/runs/run_2"):
            second_polls["n"] += 1
            if second_polls["n"] == 1:
                return _JsonResponse({"status": "RUNNING"})
            return _JsonResponse(
                {"status": "FINISHED", "result": "This is a wire out.\ndeny", "model": "composer-2"}
            )
        if method == "POST" and url.endswith("/archive"):
            return _JsonResponse({})
        raise AssertionError(f"unexpected {method} {url}")

    _install_urlopen(monkeypatch, fake_urlopen)
    results = decide_many(
        (get_fixture("core-top-up"), get_fixture("sell-all-offshore")),
        live=True,
    )
    assert [item.source for item in results] == ["live-cursor", "live-cursor"]
    assert results[0].model == "cursor-cloud-agent"
    assert results[0].parsed_action == "allow"
    assert results[1].model == "composer-2"
    assert results[1].parsed_action == "deny"
    creates = [call for call in calls if call[0] == "POST" and call[1] == "https://api.cursor.com/v1/agents"]
    followups = [call for call in calls if call[1].endswith("/agents/ag_1/runs")]
    archives = [call for call in calls if call[1].endswith("/archive")]
    assert len(creates) == 1
    assert len(followups) == 1
    assert len(archives) == 1
    assert second_polls["n"] == 2


def test_cursor_error_still_archives(monkeypatch: pytest.MonkeyPatch) -> None:
    archived = {"n": 0}

    def fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        url = request.full_url
        if request.get_method() == "POST" and url.endswith("/v1/agents"):
            return _JsonResponse({"agent": {"id": "ag_err"}, "run": {"id": "run_err"}})
        if request.get_method() == "GET":
            return _JsonResponse({"status": "ERROR", "error": "quota"})
        if url.endswith("/archive"):
            archived["n"] += 1
            return _JsonResponse({})
        raise AssertionError(url)

    _install_urlopen(monkeypatch, fake_urlopen)
    with pytest.raises(RuntimeError, match="ERROR: quota"):
        decide_llm(get_fixture("core-top-up"), live=True, api_key="crsr_test")
    assert archived["n"] == 1


def test_cursor_http_timeout_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    assert CURSOR_HTTP_TIMEOUT_SECONDS >= 120.0
    assert CURSOR_RUN_TIMEOUT_SECONDS >= 300.0
    seen: dict[str, float] = {}

    def fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        seen["timeout"] = timeout
        raise TimeoutError("The read operation timed out")

    _install_urlopen(monkeypatch, fake_urlopen)
    with pytest.raises(RuntimeError, match="about a minute") as caught:
        decide_llm(get_fixture("core-top-up"), live=True, api_key="crsr_test")
    assert seen["timeout"] == CURSOR_HTTP_TIMEOUT_SECONDS
    assert "The read operation timed out" in str(caught.value)


def test_cursor_poll_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"t": 0.0}

    def monotonic() -> float:
        clock["t"] += 100.0
        return clock["t"]

    monkeypatch.setattr("stock_policy.llm_client.time.monotonic", monotonic)

    def fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        if request.get_method() == "POST" and request.full_url.endswith("/v1/agents"):
            return _JsonResponse({"agent": {"id": "ag_slow"}, "run": {"id": "run_slow"}})
        if request.get_method() == "GET":
            return _JsonResponse({"status": "RUNNING"})
        if request.full_url.endswith("/archive"):
            return _JsonResponse({})
        raise AssertionError(request.full_url)

    _install_urlopen(monkeypatch, fake_urlopen)
    with pytest.raises(RuntimeError, match="did not finish"):
        decide_llm(get_fixture("routine-hold"), live=True, api_key="crsr_test")


def test_openai_key_stays_on_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CURSOR_API_KEY", "crsr_ignored")

    def fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        assert request.full_url == "https://api.openai.com/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer sk-test"
        return _JsonResponse({"choices": [{"message": {"content": "I would allow this."}}]})

    _install_urlopen(monkeypatch, fake_urlopen)
    result = decide_llm(get_fixture("core-top-up"), live=True)
    assert result.source == "live"
    assert result.model == "gpt-4o-mini"
    assert result.parsed_action == "allow"


def test_run_all_cursor_path_is_one_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GROK_BOT_API_KEY", "crsr_grok")
    creates = {"n": 0}
    followups = {"n": 0}

    def fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        import base64

        assert request.get_header("Authorization") == "Basic " + base64.b64encode(b"crsr_grok:").decode()
        method = request.get_method()
        url = request.full_url
        if method == "POST" and url == "https://api.cursor.com/v1/agents":
            creates["n"] += 1
            return _JsonResponse({"agent": {"id": "ag"}, "run": {"id": "run-0"}})
        if method == "POST" and url.endswith("/runs"):
            followups["n"] += 1
            return _JsonResponse({"id": f"run-{followups['n']}"})
        if method == "GET":
            return _JsonResponse({"status": "FINISHED", "result": "Inside policy.\nallow"})
        if method == "POST" and url.endswith("/archive"):
            return _JsonResponse({})
        raise AssertionError(f"{method} {url}")

    _install_urlopen(monkeypatch, fake_urlopen)
    rows = run_all(live_jev=False, live_llm=True)
    assert creates["n"] == 1
    assert followups["n"] == 5
    assert len(rows) == 6
    assert {row.llm.source for row in rows} == {"live-cursor"}
