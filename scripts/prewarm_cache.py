"""Fill build/enrich_cache with agent results so the hosted demo can replay them without new agent calls."""
from __future__ import annotations

import argparse
import asyncio
import time

from football_analytics import replay_server as replay


async def run(game_id: int, personas: list[str], languages: list[str]) -> None:
    moments = [m for m in replay.moments(game_id) if m["importance"] >= replay.ENRICH_MIN_IMPORTANCE]
    jobs = [(m, replay.profile_from({"persona": persona, "language": language})) for persona in personas for language in languages for m in moments]
    started = time.perf_counter()

    async def one(moment, profile):
        try:
            result = await replay.enrich(game_id, moment, profile)
            state = "cached" if result.get("cached") else ("ok" if result["overlay"] and not result["fallbacks"] else f"fallback {result['fallbacks']}")
        except Exception as error:
            state = f"failed {type(error).__name__}"
        print(f"{time.perf_counter() - started:6.0f}s {moment['moment_id']} {profile['persona']}/{profile['language']}: {state}", flush=True)

    await asyncio.gather(*(one(m, p) for m, p in jobs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=0)
    parser.add_argument("--personas", nargs="+", default=["casual", "analyst"])
    parser.add_argument("--languages", nargs="+", default=["en"])
    args = parser.parse_args()
    asyncio.run(run(args.game, args.personas, args.languages))


if __name__ == "__main__":
    main()
