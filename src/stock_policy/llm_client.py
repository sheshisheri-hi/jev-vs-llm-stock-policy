"""Traditional chat completion path.

The default is a deterministic stub. It answers in prose and sometimes invents
actions that are not in the policy schema. That is the contrast with Jev:
nothing in the transport stops the model from saying `approve_with_vibes`.

If you pass live=True and OPENAI_API_KEY is set, this module POSTs to an
OpenAI-compatible `/chat/completions` endpoint and parses the text best-effort.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from stock_policy.fixtures import Fixture
from stock_policy.policy import VALID_ACTIONS, ticket_state

INVENTED_ACTIONS = ("maybe_later", "approve_with_vibes", "wire_it", "yolo_buy")
_TOKEN_RE = re.compile(
    r"\b(allow|escalate_to_human|deny|maybe_later|approve_with_vibes|wire_it|yolo_buy)\b"
)


class MissingLlmKeyError(RuntimeError):
    """Raised when a live chat call was requested and OPENAI_API_KEY is unset."""


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
        "Reply with your reasoning in prose. This is not financial advice.\n\n"
        f"{state}"
    )


def call_live(fixture: Fixture, *, api_key: str, timeout: float = 30.0) -> LlmResult:
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    body = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You review stock-order policy tickets. "
                        "Answer in prose. Prefer one of allow, escalate_to_human, deny."
                    ),
                },
                {"role": "user", "content": prompt_for(fixture)},
            ],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM connection failed: {exc.reason}") from exc

    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"LLM response had no message content: {payload!r}") from exc
    if not isinstance(text, str):
        text = str(text)
    return _result("live", model, text)


def decide(fixture: Fixture, *, live: bool = False, api_key: str | None = None) -> LlmResult:
    """Stub unless `live` is True. A set OPENAI_API_KEY does not flip the default."""

    if not live:
        return _result("stub", "stub-traditional-llm", stub_text(fixture.id))
    key = (api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")).strip()
    if not key:
        raise MissingLlmKeyError(
            "Live LLM was requested, but OPENAI_API_KEY is not set. "
            "The default demo uses the stub, which needs no key."
        )
    return call_live(fixture, api_key=key)
