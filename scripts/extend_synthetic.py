import argparse
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TIMESTEPS_PER_MATCH = 3000
MATCH_MINUTES = 90
TACKLE_RATE = 0.5
PRESSURE_RATE = 0.03
TACKLE_SOURCES = ("possession_gain_candidate", "interception_candidate")
EVENT_COLUMNS = ["event_id", "GAME_ID", "start_timestep", "end_timestep", "event_type", "PLAYER_ID", "TEAM_ID", "data_source", "synthetic", "label"]
SUB_COLUMNS = ["GAME_ID", "TEAM_ID", "start_timestep", "player_off_id", "player_on_id", "data_source", "synthetic"]
REFEREES = ["Marcus Delacroix-Hale", "Ines Varga-Whitmore", "Oskar Pellinger", "Bruna Castellan"]
VENUES = ["Kestrel Vale Municipal Ground", "Saltmere Quay Arena"]
WEATHER = ["Clear, 14 C", "Light rain, 11 C", "Overcast, 12 C", "Windy, 9 C"]
KICKOFFS = ["15:00", "17:30", "19:45"]


def minute_to_timestep(minute):
    return int(round(minute / MATCH_MINUTES * TIMESTEPS_PER_MATCH))


def load_roster(path):
    players = json.loads(Path(path).read_text(encoding="utf-8"))["players"]
    club = {int(i): (1 if int(i) >= 100 else 0) for i in players}
    starters = {c: sorted(i for i, v in club.items() if v == c and not players[str(i)].get("bench")) for c in (0, 1)}
    bench = {c: sorted(i for i, v in club.items() if v == c and players[str(i)].get("bench")) for c in (0, 1)}
    return starters, bench


def build_game(game_id, events, starters, bench):
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
    subs = []
    for team in (0, 1):
        k = rng.randint(1, 3)
        offs = rng.sample(starters[team][1:], k)
        ons = rng.sample(bench[team][1:], min(k, len(bench[team]) - 1))
        for off, on in zip(offs, ons):
            ts = minute_to_timestep(rng.uniform(60, 75))
            subs.append([game_id, team, ts, off, on, "synthetic", True])
    return pd.DataFrame(rows, columns=EVENT_COLUMNS), pd.DataFrame(subs, columns=SUB_COLUMNS)


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
    starters, bench = load_roster(roster_path)
    events = pd.read_csv(Path(folder) / "events.csv")
    ev_parts, sub_parts, meta = [], [], {}
    for game_id, g in events.groupby("GAME_ID", sort=True):
        e, s = build_game(int(game_id), g, starters, bench)
        ev_parts.append(e)
        sub_parts.append(s)
        meta[str(int(game_id))] = build_meta(int(game_id))
    out = Path(out_folder)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(ev_parts).to_csv(out / "synthetic_events.csv", index=False, lineterminator="\n")
    pd.concat(sub_parts).to_csv(out / "synthetic_substitutions.csv", index=False, lineterminator="\n")
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
