"""Integration checks for the optional model commands."""
import pandas as pd
import pytest
from football_analytics.cli import main
from football_analytics.audited.xg_model import build_shot_feature_table


def test_xg_rejects_unreviewed_candidate_events():
    events = pd.DataFrame([{"GAME_ID": 0, "TIMESTEP": 1, "event_type": "shot_candidate"}])
    with pytest.raises(ValueError, match="No eligible shot events"):
        build_shot_feature_table(pd.DataFrame(), events)


@pytest.mark.parametrize("command", ["xg-train", "xg-predict", "audited-analyze"])
def test_updated_commands_are_reachable(command, capsys):
    with pytest.raises(SystemExit) as result:
        main([command, "--help"])
    assert result.value.code == 0
    assert "--config" in capsys.readouterr().out
