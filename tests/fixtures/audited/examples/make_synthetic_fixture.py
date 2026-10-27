"""Generate a clearly synthetic 22-player + ball fixture for integration tests."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def build_fixture(games: int = 2, timesteps: int = 8) -> pd.DataFrame:
    rows = []
    ball_x_by_t = [10, 11, 16, 17, 30, 40, 60, 64]
    owners_by_t = [0, 0, 1, 1, 11, None, 2, 2]
    for game in range(games):
        score0 = score1 = 0
        for t in range(timesteps):
            if t == 6:
                score0 = 1
            owner = owners_by_t[t % len(owners_by_t)]
            bx = float(ball_x_by_t[t % len(ball_x_by_t)] + game * 0.5)
            by = float(30 + (t % 3) * 2)
            for pid in range(22):
                team = 0 if pid < 11 else 1
                slot = pid if team == 0 else pid - 11
                x = (8 + (slot % 4) * 7) if team == 0 else (60 + (slot % 4) * 7)
                y = 8 + (slot // 4) * 18 + (slot % 2) * 3
                # A small, known player movement within each game.
                x = float(x + 0.2 * t * (1 if pid == 2 else 0))
                y = float(y)
                on_ball = 1 if owner is not None and pid == owner else 0
                rows.append({
                    "GAME_ID": game + 100, "TIMESTEP": t,
                    "TEAM_0_SCORE": score0, "TEAM_1_SCORE": score1,
                    "PLAYER_ID": pid, "TEAM_ID": team,
                    "PLAYER_ROLE": (["goalkeeper", "centre back", "left back", "right back", "central midfield", "central midfield", "central front", "central front", "winger", "winger", "striker"][slot]),
                    "LOCATION_X": x, "LOCATION_Y": y, "LOCATION_Z": -1,
                    "VELOCITY_X": 0.2 if pid == 2 else 0.0, "VELOCITY_Y": 0.0, "VELOCITY_Z": -1,
                    "ON_BALL": on_ball,
                })
            rows.append({
                "GAME_ID": game + 100, "TIMESTEP": t,
                "TEAM_0_SCORE": score0, "TEAM_1_SCORE": score1,
                "PLAYER_ID": -1, "TEAM_ID": -1, "PLAYER_ROLE": "ball",
                "LOCATION_X": bx, "LOCATION_Y": by, "LOCATION_Z": 0.2,
                "VELOCITY_X": (bx - ball_x_by_t[(t - 1) % len(ball_x_by_t)]) if t > 0 else 0.0,
                "VELOCITY_Y": 0.5 if t > 0 else 0.0, "VELOCITY_Z": 0.0,
                "ON_BALL": 0,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(Path(__file__).with_name("synthetic_tracking.csv")))
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=8)
    args = parser.parse_args()
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    build_fixture(args.games, args.timesteps).to_csv(dest, index=False)
    print(f"Wrote synthetic fixture to {dest} (not real match data).")

if __name__ == "__main__":
    main()
