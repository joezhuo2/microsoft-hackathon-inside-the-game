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
    for name in ("synthetic_events.csv", "synthetic_substitutions.csv"):
        for df in frames(generated, name):
            assert len(df) > 0
            assert (df["data_source"] == "synthetic").all()
            assert (df["synthetic"].astype(str) == "True").all()


def test_event_ids_unique_with_prefix(generated):
    ids = pd.concat(frames(generated, "synthetic_events.csv"))["event_id"]
    assert ids.is_unique and ids.str.startswith("SYN-").all()


def test_player_ids_in_roster(generated):
    roster = {int(i) for i in json.loads((ROOT / "data" / "roster.json").read_text(encoding="utf-8"))["players"]}
    ev = pd.concat(frames(generated, "synthetic_events.csv"))
    subs = pd.concat(frames(generated, "synthetic_substitutions.csv"))
    used = set(ev["PLAYER_ID"]) | set(subs["player_off_id"]) | set(subs["player_on_id"])
    assert used <= roster


def test_substitution_window(generated):
    subs = pd.concat(frames(generated, "synthetic_substitutions.csv"))
    assert subs["start_timestep"].between(2000, 2500).all()


def test_deterministic(generated, tmp_path):
    ext.run(ROOT / "output", tmp_path)
    for p in generated.glob("games_*/synthetic_*"):
        assert p.read_bytes() == (tmp_path / p.parent.name / p.name).read_bytes()
