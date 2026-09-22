"""Run both paths and render Input / LLM / Jev / Final gate."""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict, dataclass
from typing import Any

from stock_policy.fixtures import FIXTURES, Fixture, get_fixture
from stock_policy.gate import GateDecision, apply_gate
from stock_policy.jev_client import JevReadout, decide as decide_jev
from stock_policy.llm_client import LlmResult, decide as decide_llm
from stock_policy.policy import HardRuleResult, evaluate_hard_rules

COLUMNS = ("Input", "LLM", "Jev", "Final gate")
# Fixed widths so the demo and the README sample stay aligned.
WIDTHS = (30, 34, 40, 30)


@dataclass(frozen=True)
class Comparison:
    fixture_id: str
    title: str
    expected_final: str
    hard: HardRuleResult
    llm: LlmResult
    jev: JevReadout
    gate: GateDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "title": self.title,
            "expected_final": self.expected_final,
            "hard_rules": {"action": self.hard.action, "reasons": list(self.hard.reasons)},
            "llm": asdict(self.llm) | {"clamped_action": self.llm.clamped_action},
            "jev": asdict(self.jev),
            "final_gate": {
                "action": self.gate.action,
                "reason": self.gate.reason,
                "hard_action": self.gate.hard_action,
                "model_action": self.gate.model_action,
            },
        }


def run_fixture(
    fixture: Fixture,
    *,
    live_jev: bool | None = None,
    live_llm: bool = False,
) -> Comparison:
    hard = evaluate_hard_rules(fixture)
    llm = decide_llm(fixture, live=live_llm)
    jev = decide_jev(fixture, live=live_jev)
    gate = apply_gate(hard, jev)
    return Comparison(
        fixture_id=fixture.id,
        title=fixture.title,
        expected_final=fixture.expected_final,
        hard=hard,
        llm=llm,
        jev=jev,
        gate=gate,
    )


def run_all(*, live_jev: bool | None = None, live_llm: bool = False, fixture_id: str | None = None) -> list[Comparison]:
    if fixture_id:
        selected = (get_fixture(fixture_id),)
    else:
        selected = FIXTURES
    return [run_fixture(fixture, live_jev=live_jev, live_llm=live_llm) for fixture in selected]


def _fmt_probs(probabilities: dict[str, float]) -> str:
    parts = [f"{name}={probability:.2f}" for name, probability in probabilities.items()]
    return "{" + ", ".join(parts) + "}"


def input_cell(comparison: Comparison) -> str:
    fixture = get_fixture(comparison.fixture_id)
    order = fixture.order
    price = "mkt" if order.limit_price_usd is None else f"${order.limit_price_usd:,.0f}"
    destination = f" -> {order.destination}" if order.destination else ""
    return "\n".join(
        (
            f"{fixture.id}",
            fixture.title,
            f"{order.side.upper()} {order.quantity:g} {order.symbol} @ {price}",
            f"notional ${order.notional_usd:,.0f} | {order.session}",
            "intents: " + ", ".join(order.intents) + destination,
        )
    )


def llm_cell(comparison: Comparison) -> str:
    llm = comparison.llm
    if llm.parsed_action:
        parsed = f"parsed: {llm.parsed_action}"
    elif llm.invented_action:
        parsed = f"OUT OF SCHEMA: {llm.invented_action}"
    else:
        parsed = "OUT OF SCHEMA: unparsed"
    raw = " ".join(llm.raw_text.split())
    return "\n".join(
        (
            f"{llm.source} / {llm.model}",
            raw,
            parsed,
            f"clamp -> {llm.clamped_action}",
        )
    )


def jev_cell(comparison: Comparison) -> str:
    jev = comparison.jev
    tag = "SAMPLE" if jev.source == "sample" else "LIVE"
    lines = [
        f"{tag} model={jev.model}",
        f"action {jev.action} conf={jev.action_confidence:.2f}",
        f"p {_fmt_probs(jev.action_probabilities)}",
        f"risk {jev.risk_label} score={jev.risk_score:.2f} conf={jev.risk_confidence:.2f}",
        f"risk p {_fmt_probs(jev.risk_probabilities)}",
        f"outside_policy={jev.outside_policy:.2f}",
        f"exfil_or_fraud={jev.looks_like_exfil_or_fraud:.2f}",
    ]
    if "override" in jev.note:
        lines.append("note: authored allow, so the hard rule can override it")
    return "\n".join(lines)


def gate_cell(comparison: Comparison) -> str:
    gate = comparison.gate
    if comparison.hard.reasons:
        rules = "rules: " + "; ".join(comparison.hard.reasons)
    else:
        rules = "rules: pass"
    return "\n".join((gate.action, rules, gate.reason))


def _wrap_block(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        wrapped = textwrap.wrap(
            paragraph,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])
    return lines


def _border(left: str, mid: str, right: str, fill: str) -> str:
    parts = [fill * (width + 2) for width in WIDTHS]
    return left + mid.join(parts) + right


def render_table(comparisons: list[Comparison]) -> str:
    """Four columns: Input, LLM, Jev, Final gate."""

    rows: list[str] = []
    rows.append(_border("┌", "┬", "┐", "─"))
    header_lines = [_wrap_block(title, width) for title, width in zip(COLUMNS, WIDTHS)]
    height = max(len(lines) for lines in header_lines)
    for index in range(height):
        cells = []
        for lines, width in zip(header_lines, WIDTHS):
            text = lines[index] if index < len(lines) else ""
            cells.append(" " + text.ljust(width) + " ")
        rows.append("│" + "│".join(cells) + "│")
    for comparison in comparisons:
        rows.append(_border("├", "┼", "┤", "─"))
        blocks = [
            _wrap_block(input_cell(comparison), WIDTHS[0]),
            _wrap_block(llm_cell(comparison), WIDTHS[1]),
            _wrap_block(jev_cell(comparison), WIDTHS[2]),
            _wrap_block(gate_cell(comparison), WIDTHS[3]),
        ]
        body_height = max(len(block) for block in blocks)
        for index in range(body_height):
            cells = []
            for block, width in zip(blocks, WIDTHS):
                text = block[index] if index < len(block) else ""
                cells.append(" " + text.ljust(width) + " ")
            rows.append("│" + "│".join(cells) + "│")
    rows.append(_border("└", "┴", "┘", "─"))
    return "\n".join(rows)


def render_report(comparisons: list[Comparison], *, jev_mode: str, llm_mode: str) -> str:
    banner = [
        "jev vs llm — stock order policy",
        "Order-policy guardrails on synthetic tickets. Not financial advice. Not a return forecast.",
        "Jev 'cannot hallucinate' here means the Choice/Noul/Score stay inside the schema you sent.",
        f"LLM path: {llm_mode}. Jev path: {jev_mode}",
        "Final gate uses hard rules + Jev only. The LLM column is the contrast; its clamp is not the decision.",
        "",
        render_table(comparisons),
        "",
    ]
    return "\n".join(banner)


def dump_json(comparisons: list[Comparison]) -> str:
    return json.dumps([item.to_dict() for item in comparisons], indent=2)
