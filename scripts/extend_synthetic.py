import argparse
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TACKLE_RATE = 0.5
PRESSURE_RATE = 0.03
TACKLE_SOURCES = ("possession_gain_candidate", "interception_candidate")
EVENT_COLUMNS = ["event_id", "GAME_ID", "start_timestep", "end_timestep", "event_type", "PLAYER_ID", "TEAM_ID", "data_source", "synthetic", "label"]
REFEREES = ["Marcus Delacroix-Hale", "Ines Varga-Whitmore", "Oskar Pellinger", "Bruna Castellan"]
VENUES = ["Kestrel Vale Municipal Ground", "Saltmere Quay Arena"]
WEATHER = ["Clear, 14 C", "Light rain, 11 C", "Overcast, 12 C", "Windy, 9 C"]
KICKOFFS = ["15:00", "17:30", "19:45"]


def load_roster(path):
    players = json.loads(Path(path).read_text(encoding="utf-8"))["players"]
    club = {int(i): (1 if int(i) >= 100 else 0) for i in players}
    return {c: sorted(i for i, v in club.items() if v == c and not players[str(i)].get("bench")) for c in (0, 1)}


def build_game(game_id, events, starters):
    rng = random.Random(f"synthetic-{game_id}")
    rows = []

    def add(row, kind, label, player, team):
        rows.append([f"SYN-{game_id}-{len(rows) + 1:05d}", game_id, int(row.start_timestep), int(row.end_timestep), kind, player, team, "synthetic", True, label])

    for row in events.itertuples():
        if row.event_type in TACKLE_SOURCES and pd.notna(row.PLAYER_ID) and rng.random() < TACKLE_RATE:
            add(row, "tackle_synthetic", "Synthetic tackle", int(row.PLAYER_ID), int(row.TEAM_ID))
        if pd.notna(row.TEAM_ID) and rng.random() < PRESSURE_RATE:
            opp = 1 - int(row.TEAM_ID)
            add(row, "pressure_synthetic", "Synthetic pressure", rng.choice(starters[opp]), opp)
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def build_meta(game_id):
    rng = random.Random(f"synthetic-meta-{game_id}")
    return {
        "referee": {"value": rng.choice(REFEREES), "synthetic": True},
        "venue": {"value": rng.choice(VENUES), "synthetic": True},
        "attendance": {"value": rng.randint(8000, 32000), "synthetic": True},
        "weather": {"value": rng.choice(WEATHER), "synthetic": True},
        "kickoff_local_time": {"value": rng.choice(KICKOFFS), "synthetic": True},
    }


def process_folder(folder, out_folder, roster_path):
    starters = load_roster(roster_path)
    events = pd.read_csv(Path(folder) / "events.csv")
    ev_parts, meta = [], {}
    for game_id, g in events.groupby("GAME_ID", sort=True):
        ev_parts.append(build_game(int(game_id), g, starters))
        meta[str(int(game_id))] = build_meta(int(game_id))
    out = Path(out_folder)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(ev_parts).to_csv(out / "synthetic_events.csv", index=False, lineterminator="\n")
    (out / "synthetic_substitutions.csv").unlink(missing_ok=True)
    payload = {"note": "All fields synthetic; not from a real match.", "synthetic": True, "games": meta}
    (out / "synthetic_match_meta.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="")


def run(output, dest=None, roster_path=ROOT / "data" / "roster.json"):
    output = Path(output)
    for folder in sorted(output.glob("games_*")):
        process_folder(folder, Path(dest) / folder.name if dest else folder, roster_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output")
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
