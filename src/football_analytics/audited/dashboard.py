"""Optional Streamlit viewer for precomputed output CSVs.

Run with: streamlit run src/football_analytics/dashboard.py -- output_directory
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - depends on optional installation
    raise SystemExit("Install the optional dashboard dependency with: pip install -e '.[dashboard]'") from exc


def main() -> None:  # pragma: no cover - UI smoke tests require Streamlit runtime
    st.set_page_config(page_title="Football Tracking Analytics", layout="wide")
    st.title("Football Tracking Analytics")
    st.caption("Candidate events and uncalibrated metrics are explicitly labelled; this dashboard does not verify inferred events.")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    output_dir = Path(args[-1]) if args else Path("output")
    frame_path = output_dir / "frame_metrics.csv"
    event_path = output_dir / "events.csv"
    if not frame_path.exists():
        st.error(f"Could not find {frame_path}. Run the CLI analysis first.")
        st.stop()
    frames = pd.read_csv(frame_path, low_memory=False)
    events = pd.read_csv(event_path, low_memory=False) if event_path.exists() else pd.DataFrame()
    game_ids = frames["GAME_ID"].dropna().unique().tolist()
    game_id = st.selectbox("Game", game_ids)
    game = frames[frames["GAME_ID"].eq(game_id)]
    timesteps = sorted(game["TIMESTEP"].dropna().unique().tolist())
    selected_timestep = st.select_slider("Frame / timestep", options=timesteps)
    before, after = st.columns(2)
    with before:
        st.metric("Entities in frame", int((game["TIMESTEP"] == selected_timestep).sum()))
    with after:
        current_events = events[(events["GAME_ID"].eq(game_id)) & (events["TIMESTEP"].between(selected_timestep - 1, selected_timestep + 1))] if not events.empty else pd.DataFrame()
        st.metric("Candidate events nearby", int(len(current_events)))
    snapshot = game[game["TIMESTEP"].eq(selected_timestep)].copy()
    snapshot["plot_entity"] = snapshot.apply(lambda r: "Ball" if r.get("ENTITY_TYPE") == "ball" else f"Team {r.get('TEAM_ID')} · {r.get('PLAYER_ROLE', 'player')}", axis=1)
    valid = snapshot.dropna(subset=["LOCATION_X", "LOCATION_Y"])
    if valid.empty:
        st.warning("No valid x/y positions at this frame.")
    else:
        fig = px.scatter(valid, x="LOCATION_X", y="LOCATION_Y", color="plot_entity", hover_data=[c for c in ["PLAYER_ID", "TEAM_ID", "PLAYER_ROLE", "SPEED_NATIVE_SOURCE", "DISTANCE_TO_BALL_NATIVE", "possession_state"] if c in valid.columns], title=f"Game {game_id}, timestep {selected_timestep}")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(legend_title="Entity", height=650)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Nearby candidate events")
    if current_events.empty:
        st.info("No detected candidate events within one timestep of this frame.")
    else:
        cols = [c for c in ["TIMESTEP", "event_type", "TEAM_ID", "PLAYER_ID", "RELATED_PLAYER_ID", "confidence", "event_outcome", "evidence_json", "alternative_interpretations"] if c in current_events.columns]
        st.dataframe(current_events[cols], use_container_width=True, hide_index=True)
    st.subheader("Player movement snapshot")
    players = snapshot[snapshot["ENTITY_TYPE"].eq("player")]
    show_cols = [c for c in ["PLAYER_ID", "TEAM_ID", "PLAYER_ROLE", "SPEED_NATIVE_SOURCE", "DISTANCE_TO_BALL_NATIVE", "NEAREST_TEAMMATE_DISTANCE_NATIVE", "NEAREST_OPPONENT_DISTANCE_NATIVE", "possession_state"] if c in players.columns]
    st.dataframe(players[show_cols], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
