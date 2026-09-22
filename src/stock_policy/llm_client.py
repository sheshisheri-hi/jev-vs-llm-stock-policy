"""Traditional chat path.

The default is a deterministic stub. It answers in prose and sometimes invents
actions that are not in the policy schema. That is the contrast with Jev:
nothing in the transport stops the model from saying `approve_with_vibes`.

`live=True` picks a backend from the key:

- A normal `OPENAI_API_KEY` (not prefixed `crsr_`) uses an OpenAI-compatible
  `/chat/completions` endpoint.
- `GROK_BOT_API_KEY`, `CURSOR_API_KEY`, or any `crsr_` key uses the Cursor
  Cloud Agents API. Cursor does not speak `/v1/chat/completions`. One no-repo
  agent handles the whole run: create, then a follow-up run per extra ticket,
  then archive.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass

from stock_policy.fixtures import Fixture
from stock_policy.policy import VALID_ACTIONS, ticket_state

INVENTED_ACTIONS = ("maybe_later", "approve_with_vibes", "wire_it", "yolo_buy")
_TOKEN_RE = re.compile(
    r"\b(allow|escalate_to_human|deny|maybe_later|approve_with_vibes|wire_it|yolo_buy)\b"
)


class MissingLlmKeyError(RuntimeError):
    """Raised when a live LLM call was requested and no usable key is set."""


CURSOR_API_ROOT = "https://api.cursor.com/v1"
CURSOR_AGENT_NAME = "jev-demo-llm"
CURSOR_SOURCE = "live-cursor"
CURSOR_MODEL = "cursor-cloud-agent"
CURSOR_RUN_TIMEOUT_SECONDS = 180.0
CURSOR_HTTP_TIMEOUT_SECONDS = 30.0
_MISSING_LLM_KEY = (
    "Live LLM was requested, but no key is set. "
    "Set GROK_BOT_API_KEY or CURSOR_API_KEY for a Cursor Cloud Agent, "
    "or OPENAI_API_KEY for an OpenAI-compatible chat model. "
    "The default demo uses the stub, which needs no key."
)


@dataclass(frozen=True)
class LlmParse:
    parsed_action: str | None
    invented_action: str | None
    note: str

    @property
    def clamped_action(self) -> str:
        """Fail closed. An unusable free-form answer becomes escalate, not allow."""

        if self.parsed_action in VALID_ACTIONS:
            return self.parsed_action
        return "escalate_to_human"


@dataclass(frozen=True)
class LlmResult:
    source: str
    model: str
    raw_text: str
    parsed_action: str | None
    invented_action: str | None
    note: str

    @property
    def clamped_action(self) -> str:
        if self.parsed_action in VALID_ACTIONS:
            return self.parsed_action
        return "escalate_to_human"


def parse_llm_text(text: str) -> LlmParse:
    """Accept only allow, escalate_to_human, or deny. Anything else is not a decision."""

    json_action = _json_action(text)
    if json_action is not None:
        if json_action in VALID_ACTIONS:
            return LlmParse(json_action, None, "parsed a schema action from JSON")
        return LlmParse(
            None,
            json_action,
            f"rejected out-of-schema JSON action {json_action!r}",
        )

    found = _TOKEN_RE.findall(text)
    invented = [token for token in found if token not in VALID_ACTIONS]
    valid = [token for token in found if token in VALID_ACTIONS]
    if invented and not valid:
        return LlmParse(None, invented[0], f"rejected invented action {invented[0]!r}")
    if invented and valid:
        return LlmParse(
            None,
            invented[0],
            "prose mixed a schema word with an invented action; rejected",
        )
    unique = list(dict.fromkeys(valid))
    if len(unique) == 1:
        return LlmParse(unique[0], None, "parsed a schema action from prose")
    if len(unique) > 1:
        return LlmParse(None, None, "prose named more than one schema action; rejected")
    return LlmParse(None, None, "no schema action found in free-form text")


def _json_action(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        payload, _end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("action", "decision"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


_STUBS: dict[str, str] = {
    "core-top-up": (
        "Yeah I'd allow this. Tiny CORE add, nothing spicy. "
        '{"action": "allow", "confidence": "pretty sure", "vibes": "green"}'
    ),
    "routine-hold": (
        "No trade here. I might maybe_later if the PM pings again, "
        "but for now I'm not even sure this is an order. Thoughts?"
    ),
    "concentration-breach": (
        "Concentration is a guideline more than a law. "
        "I'll approve_with_vibes and we can tidy the sleeve next week."
    ),
    "after-hours-options": (
        "Delta is just a number. maybe_later when the open prints, "
        "or we yolo_buy the calls tonight if the client is impatient."
    ),
    "sell-all-offshore": (
        "They sound rushed but also kind of convincing? "
        '{"decision": "wire_it", "note": "client insisted, so probably fine"}'
    ),
    "gray-zone-tip": (
        "Hmm the note says rebalance but also says you didn't hear this from me. "
        "I'll go with approve_with_vibes and mention it in standup. "
        "Could also deny. Or not."
    ),
}


def stub_text(fixture_id: str) -> str:
    try:
        return _STUBS[fixture_id]
    except KeyError as exc:
        raise KeyError(f"No LLM stub for {fixture_id!r}") from exc


def _result(source: str, model: str, raw_text: str) -> LlmResult:
    parsed = parse_llm_text(raw_text)
    return LlmResult(
        source=source,
        model=model,
        raw_text=raw_text,
        parsed_action=parsed.parsed_action,
        invented_action=parsed.invented_action,
        note=parsed.note,
    )


def prompt_for(fixture: Fixture) -> str:
    state = json.dumps(ticket_state(fixture), indent=2)
    return (
        "You are a broker-dealer order-policy reviewer. "
        "Decide whether this synthetic ticket should be allow, escalate_to_human, or deny. "
        "Reply with your reasoning in prose. "
        "End with a final line that is exactly one of: allow, escalate_to_human, deny. "
        "Do not name the other actions in the prose. "
        "This is not financial advice.\n\n"
        f"{state}"
    )


def resolve_llm_credentials(explicit: str | None = None) -> tuple[str | None, str]:
    """Return `(key, backend)` where backend is `openai`, `cursor`, or `missing`.

    Order: a non-`crsr_` `OPENAI_API_KEY`, else `GROK_BOT_API_KEY`, else
    `CURSOR_API_KEY`, else an `OPENAI_API_KEY` that itself starts with `crsr_`.
    Keys from the Cursor variables use Cloud Agents even when they lack the prefix.
    An explicit key uses Cloud Agents only when it starts with `crsr_`.
    """

    if explicit is not None:
        key = explicit.strip()
        if not key:
            return None, "missing"
        if key.startswith("crsr_"):
            return key, "cursor"
        return key, "openai"

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key and not openai_key.startswith("crsr_"):
        return openai_key, "openai"
    grok_key = os.environ.get("GROK_BOT_API_KEY", "").strip()
    if grok_key:
        return grok_key, "cursor"
    cursor_key = os.environ.get("CURSOR_API_KEY", "").strip()
    if cursor_key:
        return cursor_key, "cursor"
    if openai_key.startswith("crsr_"):
        return openai_key, "cursor"
    return None, "missing"


def live_llm_ready() -> bool:
    key, _backend = resolve_llm_credentials()
    return key is not None


def describe_live_llm() -> str:
    _key, backend = resolve_llm_credentials()
    if backend == "cursor":
        return "live Cursor Cloud Agent (no repo; one agent for the run, then archive)"
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    return f"live OpenAI-compatible chat ({model})"


def _basic_auth(api_key: str) -> str:
    token = base64.b64encode(f"{api_key}:".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def _read_response(response: object) -> dict[str, object]:
    raw = response.read().decode("utf-8")  # type: ignore[attr-defined]
    if not raw.strip():
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError(f"LLM response was not a JSON object: {payload!r}")
    return payload


def _http_json(
    method: str,
    url: str,
    *,
    body: dict[str, object] | None,
    timeout: float,
    auth: str,
) -> dict[str, object]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Authorization": auth, "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _read_response(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"LLM HTTP {exc.code} {method} {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM connection failed: {exc.reason}") from exc


def _call_openai(fixture: Fixture, *, api_key: str, timeout: float = 30.0) -> LlmResult:
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    payload = _http_json(
        "POST",
        f"{base}/chat/completions",
        body={
            "model": model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You review stock-order policy tickets. "
                        "Answer in prose. End with a final line that is only the chosen action: "
                        "allow, escalate_to_human, or deny."
                    ),
                },
                {"role": "user", "content": prompt_for(fixture)},
            ],
        },
        timeout=timeout,
        auth=f"Bearer {api_key}",
    )
    try:
        text = payload["choices"][0]["message"]["content"]  # type: ignore[index]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"LLM response had no message content: {payload!r}") from exc
    if not isinstance(text, str):
        text = str(text)
    return _result("live", model, text)


def _nested_id(payload: dict[str, object], key: str) -> str | None:
    block = payload.get(key)
    if isinstance(block, dict):
        value = block.get("id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(block, str) and block.strip():
        return block.strip()
    return None


def _run_status(payload: dict[str, object]) -> tuple[str, str | None, str]:
    status = payload.get("status")
    result = payload.get("result")
    model = payload.get("model") if isinstance(payload.get("model"), str) else None
    run = payload.get("run")
    if isinstance(run, dict):
        if not isinstance(status, str):
            status = run.get("status")
        if not isinstance(result, str):
            result = run.get("result")
        if model is None and isinstance(run.get("model"), str):
            model = run["model"]
    status_text = str(status or "").upper()
    text = result if isinstance(result, str) else None
    return status_text, text, model or CURSOR_MODEL


def _error_detail(payload: dict[str, object]) -> str:
    detail = payload.get("error") or payload.get("message")
    if isinstance(detail, dict):
        message = detail.get("message")
        return str(message) if message else json.dumps(detail)
    return "" if detail is None else str(detail)


class CursorAgentSession:
    """One no-repo Cloud Agent. Create on the first ticket, follow-up runs after that."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self.agent_id: str | None = None

    def complete(self, fixture: Fixture) -> LlmResult:
        prompt = prompt_for(fixture)
        if self.agent_id is None:
            self.agent_id, run_id = self._create(prompt)
        else:
            run_id = self._follow_up(prompt)
        text, model = self._poll(self.agent_id, run_id)
        return _result(CURSOR_SOURCE, model, text)

    def archive(self) -> None:
        if self.agent_id is None:
            return
        agent_id = self.agent_id
        self.agent_id = None
        _http_json(
            "POST",
            f"{CURSOR_API_ROOT}/agents/{agent_id}/archive",
            body={},
            timeout=CURSOR_HTTP_TIMEOUT_SECONDS,
            auth=_basic_auth(self._api_key),
        )

    def _create(self, prompt: str) -> tuple[str, str]:
        # No repos and no env: this agent has nothing to clone or inject.
        payload = _http_json(
            "POST",
            f"{CURSOR_API_ROOT}/agents",
            body={"prompt": {"text": prompt}, "name": CURSOR_AGENT_NAME},
            timeout=CURSOR_HTTP_TIMEOUT_SECONDS,
            auth=_basic_auth(self._api_key),
        )
        agent_id = _nested_id(payload, "agent")
        run_id = _nested_id(payload, "run")
        if not agent_id or not run_id:
            raise RuntimeError(f"Cursor create response had no agent.id and run.id: {payload!r}")
        return agent_id, run_id

    def _follow_up(self, prompt: str) -> str:
        assert self.agent_id is not None
        payload = _http_json(
            "POST",
            f"{CURSOR_API_ROOT}/agents/{self.agent_id}/runs",
            body={"prompt": {"text": prompt}},
            timeout=CURSOR_HTTP_TIMEOUT_SECONDS,
            auth=_basic_auth(self._api_key),
        )
        run_id = _nested_id(payload, "run")
        if run_id is None:
            top = payload.get("id")
            if isinstance(top, str) and top.strip():
                run_id = top.strip()
        if not run_id:
            raise RuntimeError(f"Cursor follow-up response had no run id: {payload!r}")
        return run_id

    def _poll(self, agent_id: str, run_id: str) -> tuple[str, str]:
        deadline = time.monotonic() + CURSOR_RUN_TIMEOUT_SECONDS
        delay = 1.0
        last_status = "UNKNOWN"
        for _poll in range(10_000):
            payload = _http_json(
                "GET",
                f"{CURSOR_API_ROOT}/agents/{agent_id}/runs/{run_id}",
                body=None,
                timeout=CURSOR_HTTP_TIMEOUT_SECONDS,
                auth=_basic_auth(self._api_key),
            )
            status, text, model = _run_status(payload)
            last_status = status or last_status
            if status == "FINISHED":
                if not text or not text.strip():
                    raise RuntimeError(f"Cursor agent run {run_id} finished with an empty result")
                return text, model
            if status in {"ERROR", "CANCELLED"}:
                detail = _error_detail(payload)
                extra = f": {detail}" if detail else ""
                raise RuntimeError(f"Cursor agent run {run_id} {status}{extra}")
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Cursor agent run {run_id} did not finish within "
                    f"{CURSOR_RUN_TIMEOUT_SECONDS:.0f}s (last status {last_status})"
                )
            time.sleep(delay)
            delay = min(delay * 1.5, 8.0)
        raise RuntimeError(f"Cursor agent run {run_id} exceeded the poll guard (last status {last_status})")


def _cursor_many(fixtures: Sequence[Fixture], api_key: str) -> list[LlmResult]:
    session = CursorAgentSession(api_key)
    results: list[LlmResult] = []
    run_error: Exception | None = None
    try:
        for fixture in fixtures:
            results.append(session.complete(fixture))
    except Exception as exc:
        run_error = exc
    archive_error: Exception | None = None
    try:
        session.archive()
    except Exception as exc:
        archive_error = exc
    if run_error is not None and archive_error is not None:
        raise RuntimeError(f"{run_error}; also failed to archive the Cursor agent: {archive_error}") from run_error
    if run_error is not None:
        raise run_error
    if archive_error is not None:
        raise archive_error
    return results


def decide_many(
    fixtures: Sequence[Fixture],
    *,
    live: bool = False,
    api_key: str | None = None,
) -> list[LlmResult]:
    """Stub each ticket, or one live backend for the whole sequence.

    A Cursor key reuses a single no-repo agent across `fixtures`.
    """

    if not live:
        return [_result("stub", "stub-traditional-llm", stub_text(fixture.id)) for fixture in fixtures]
    key, backend = resolve_llm_credentials(api_key)
    if not key:
        raise MissingLlmKeyError(_MISSING_LLM_KEY)
    if backend == "cursor":
        return _cursor_many(fixtures, key)
    return [_call_openai(fixture, api_key=key) for fixture in fixtures]


def decide(fixture: Fixture, *, live: bool = False, api_key: str | None = None) -> LlmResult:
    """Stub unless `live` is True. A set key does not flip the default."""

    return decide_many((fixture,), live=live, api_key=api_key)[0]


def call_live(fixture: Fixture, *, api_key: str, timeout: float = 30.0) -> LlmResult:
    """One live ticket. `crsr_` keys go to Cloud Agents; anything else to chat completions."""

    if api_key.startswith("crsr_"):
        return decide(fixture, live=True, api_key=api_key)
    return _call_openai(fixture, api_key=api_key, timeout=timeout)
