#!/usr/bin/env python3
"""Print the six order-policy tickets: Input, LLM, Jev, Final gate.

Exit 0 with the LLM stub and authored Jev SAMPLE answers when no key is set.
A live Jev call is used automatically when TYPESAFE_API_KEY or JEV_API_KEY is set.
Pass --live-jev or --live-llm to require the corresponding key.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from stock_policy.compare import dump_json, render_report, run_all  # noqa: E402
from stock_policy.fixtures import FIXTURES  # noqa: E402
from stock_policy.jev_client import MissingJevKeyError, resolve_api_key  # noqa: E402
from stock_policy.llm_client import MissingLlmKeyError  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a free-form LLM with Jev on stock order policy.")
    parser.add_argument("--live-jev", action="store_true", help="Require a live System One call (jev-latest).")
    parser.add_argument("--live-llm", action="store_true", help="Call an OpenAI-compatible chat model.")
    parser.add_argument("--fixture", help="Run one fixture id instead of all six.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the table.")
    return parser


def _known_ids() -> str:
    return ", ".join(fixture.id for fixture in FIXTURES)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    args = _parser().parse_args(argv)
    if args.fixture:
        known = {fixture.id for fixture in FIXTURES}
        if args.fixture not in known:
            print(f"Unknown fixture {args.fixture!r}. Known ids: {_known_ids()}", file=sys.stderr)
            return 2

    jev_key = resolve_api_key()
    if args.live_jev and not jev_key:
        print(
            "No TYPESAFE_API_KEY or JEV_API_KEY. Copy .env.example to .env and set one of them, "
            "or drop --live-jev to print authored SAMPLE answers.",
            file=sys.stderr,
        )
        return 2
    if args.live_llm and not os.environ.get("OPENAI_API_KEY", "").strip():
        print(
            "No OPENAI_API_KEY. The default path is the messy stub and needs no key. "
            "Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL / OPENAI_MODEL) to use --live-llm.",
            file=sys.stderr,
        )
        return 2

    live_jev: bool | None = True if args.live_jev else None
    if jev_key and not args.live_jev:
        jev_mode = "live jev-latest (key detected; System One)."
    elif args.live_jev:
        jev_mode = "live jev-latest (System One)."
    else:
        jev_mode = (
            "SAMPLE authored answers — set TYPESAFE_API_KEY or JEV_API_KEY to call jev-latest. "
            "SAMPLE rows are schema-valid illustrations, not live model output."
        )
    if args.live_llm:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        llm_mode = f"live OpenAI-compatible chat ({model})"
    else:
        llm_mode = "stub traditional LLM (free-form text; pass --live-llm for a real chat model)"

    try:
        comparisons = run_all(live_jev=live_jev, live_llm=args.live_llm, fixture_id=args.fixture)
    except MissingJevKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except MissingLlmKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Call failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(dump_json(comparisons))
        print(f"LLM path: {llm_mode}", file=sys.stderr)
        print(f"Jev path: {jev_mode}", file=sys.stderr)
        return 0

    print(render_report(comparisons, jev_mode=jev_mode, llm_mode=llm_mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
