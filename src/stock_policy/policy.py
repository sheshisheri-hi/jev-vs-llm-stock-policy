"""Hard limits evaluated in code. These beat a model that says allow.

Arithmetic lives here on purpose. Jev is a judgment model; notional,
concentration, delta, and the session clock are exact checks.
"""

from __future__ import annotations

from dataclasses import dataclass

from stock_policy.fixtures import Fixture

# Synthetic desk policy. Educational numbers, not a real mandate.
MAX_ORDER_NOTIONAL_USD = 250_000.0
MAX_CONCENTRATION_PCT = 15.0
OPTIONS_DELTA_HUMAN_ABOVE = 0.40
AFTER_HOURS_AGGRESSIVE_NOTIONAL_USD = 25_000.0
BLOCKED_INTENTS = frozenset({"wire_out", "offshore_transfer", "sell_all"})

VALID_ACTIONS = ("allow", "escalate_to_human", "deny")


@dataclass(frozen=True)
class OrderFacts:
    order_notional_usd: float
    current_concentration_pct: float | None
    projected_concentration_pct: float | None
    abs_delta: float | None
    session: str
    intents: tuple[str, ...]
    multi_party_auth: bool


@dataclass(frozen=True)
class HardRuleResult:
    """`action` is the floor the gate must respect. None means the rules passed."""

    action: str | None
    reasons: tuple[str, ...]


def compute_facts(fixture: Fixture) -> OrderFacts:
    order = fixture.order
    book = fixture.portfolio_value_usd
    current_value = sum(
        position.market_value_usd
        for position in fixture.positions
        if position.symbol == order.symbol
    )
    current_pct = (100.0 * current_value / book) if book > 0 else None

    if order.side == "buy":
        next_value = current_value + order.notional_usd
    elif order.side == "sell":
        next_value = max(0.0, current_value - order.notional_usd)
    else:
        next_value = current_value
    projected_pct = (100.0 * next_value / book) if book > 0 else None
    # A sell-all / book-level wire is not a single-name concentration event.
    if order.symbol == "ALL" or "sell_all" in order.intents:
        projected_pct = None
        current_pct = None

    return OrderFacts(
        order_notional_usd=order.notional_usd,
        current_concentration_pct=current_pct,
        projected_concentration_pct=projected_pct,
        abs_delta=None if order.delta is None else abs(order.delta),
        session=order.session,
        intents=order.intents,
        multi_party_auth=order.multi_party_auth,
    )


def policy_card() -> dict[str, object]:
    """The written limits, included in model state so the question can point at them."""

    return {
        "max_order_notional_usd": MAX_ORDER_NOTIONAL_USD,
        "max_position_concentration_pct": MAX_CONCENTRATION_PCT,
        "options_delta_requires_human_above": OPTIONS_DELTA_HUMAN_ABOVE,
        "after_hours_escalate_notional_usd": AFTER_HOURS_AGGRESSIVE_NOTIONAL_USD,
        "blocked_without_multi_party_auth": sorted(BLOCKED_INTENTS),
        "actions": list(VALID_ACTIONS),
    }


def evaluate_hard_rules(fixture: Fixture) -> HardRuleResult:
    """Return deny, escalate_to_human, or no floor.

    Deny outranks escalate. A passing result has action None.
    """

    facts = compute_facts(fixture)
    order = fixture.order
    denies: list[str] = []
    escalates: list[str] = []

    blocked = sorted(set(order.intents) & BLOCKED_INTENTS)
    if blocked and not order.multi_party_auth:
        denies.append(
            "blocklist "
            + ", ".join(blocked)
            + " without multi-party auth"
        )

    if order.side != "hold" and facts.order_notional_usd > MAX_ORDER_NOTIONAL_USD:
        denies.append(
            f"notional ${facts.order_notional_usd:,.0f} exceeds "
            f"${MAX_ORDER_NOTIONAL_USD:,.0f}"
        )

    projected = facts.projected_concentration_pct
    if (
        order.side == "buy"
        and projected is not None
        and projected > MAX_CONCENTRATION_PCT
    ):
        escalates.append(
            f"projected concentration {projected:.1f}% exceeds "
            f"{MAX_CONCENTRATION_PCT:.0f}%"
        )

    if (
        order.asset_class == "option"
        and facts.abs_delta is not None
        and facts.abs_delta > OPTIONS_DELTA_HUMAN_ABOVE
    ):
        escalates.append(
            f"option |delta| {facts.abs_delta:.2f} exceeds "
            f"{OPTIONS_DELTA_HUMAN_ABOVE:.2f}; a human must release it"
        )

    if (
        order.session == "after_hours"
        and order.side != "hold"
        and facts.order_notional_usd >= AFTER_HOURS_AGGRESSIVE_NOTIONAL_USD
    ):
        escalates.append(
            f"after-hours notional ${facts.order_notional_usd:,.0f} exceeds "
            f"${AFTER_HOURS_AGGRESSIVE_NOTIONAL_USD:,.0f}"
        )

    if denies:
        return HardRuleResult("deny", tuple(denies + escalates))
    if escalates:
        return HardRuleResult("escalate_to_human", tuple(escalates))
    return HardRuleResult(None, ())


def ticket_state(fixture: Fixture) -> dict[str, object]:
    """Structured state for Jev. Facts are precomputed so the model is not asked to multiply."""

    facts = compute_facts(fixture)
    order = fixture.order
    return {
        "desk": "synthetic order-policy demo",
        "policy": policy_card(),
        "portfolio": {
            "value_usd": fixture.portfolio_value_usd,
            "positions": [
                {
                    "symbol": position.symbol,
                    "market_value_usd": position.market_value_usd,
                    "asset_class": position.asset_class,
                }
                for position in fixture.positions
            ],
        },
        "order": {
            "side": order.side,
            "symbol": order.symbol,
            "quantity": order.quantity,
            "limit_price_usd": order.limit_price_usd,
            "notional_usd": order.notional_usd,
            "asset_class": order.asset_class,
            "delta": order.delta,
            "session": order.session,
            "intents": list(order.intents),
            "multi_party_auth": order.multi_party_auth,
            "destination": order.destination,
        },
        "computed": {
            "order_notional_usd": facts.order_notional_usd,
            "current_concentration_pct": facts.current_concentration_pct,
            "projected_concentration_pct": facts.projected_concentration_pct,
            "abs_delta": facts.abs_delta,
            "session": facts.session,
        },
        "agent_rationale": fixture.rationale,
    }
