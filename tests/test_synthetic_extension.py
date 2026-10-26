import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("extend_synthetic", ROOT / "scripts" / "extend_synthetic.py")
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    dest = tmp_path_factory.mktemp("syn")
    ext.run(ROOT / "output", dest)
    return dest


def frames(dest, name):
    return [pd.read_csv(p) for p in sorted(dest.glob(f"games_*/{name}"))]


def test_rows_are_flagged_synthetic(generated):
    for df in frames(generated, "synthetic_events.csv"):
        assert len(df) > 0
        assert (df["data_source"] == "synthetic").all()
        assert (df["synthetic"].astype(str) == "True").all()


def test_event_ids_unique_with_prefix(generated):
    ids = pd.concat(frames(generated, "synthetic_events.csv"))["event_id"]
    assert ids.is_unique and ids.str.startswith("SYN-").all()


def test_player_ids_in_roster(generated):
    players = json.loads((ROOT / "data" / "roster.json").read_text(encoding="utf-8"))["players"]
    starters = {int(i) for i, person in players.items() if not person.get("bench")}
    assert set(pd.concat(frames(generated, "synthetic_events.csv"))["PLAYER_ID"]) <= starters


def test_no_synthetic_substitutions_against_fixed_tracking_lineup(tmp_path):
    stale = tmp_path / "games_0-2" / "synthetic_substitutions.csv"
    stale.parent.mkdir()
    stale.write_text("stale")
    ext.run(ROOT / "output", tmp_path)
    assert not list(tmp_path.glob("games_*/synthetic_substitutions.csv"))


def test_deterministic(generated, tmp_path):
    ext.run(ROOT / "output", tmp_path)
    for p in generated.glob("games_*/synthetic_*"):
        assert p.read_bytes() == (tmp_path / p.parent.name / p.name).read_bytes()
