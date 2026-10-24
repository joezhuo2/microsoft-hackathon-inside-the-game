from __future__ import annotations

import numpy as np
import pandas as pd


def infer_possession(frame: pd.DataFrame) -> pd.DataFrame:
    """Infer a candidate owner and team for each frame using ON_BALL and the nearest player as a fallback."""
    work = frame.copy()
    needed = {"GAME_ID", "TIMESTEP", "PLAYER_ID", "TEAM_ID", "LOCATION_X", "LOCATION_Y", "ON_BALL"}
    missing = needed - set(work.columns)
    if missing:
        raise ValueError(f"Missing required possession columns: {sorted(missing)}")

    work["owner_player_id"] = np.nan
    work["owner_team_id"] = np.nan
    work["ownership_source"] = "unknown"
    work["ownership_confidence"] = "unresolved"

    for game_id, game_frame in work.groupby("GAME_ID", sort=False):
        observed_distances = []
        for _, observed_frame in game_frame.groupby("TIMESTEP", sort=False):
            ball_rows = observed_frame[observed_frame["PLAYER_ID"] == -1]
            observed_players = observed_frame[
                (observed_frame["PLAYER_ID"] != -1)
                & observed_frame["ON_BALL"].fillna(False).astype(bool)
            ]
            if not ball_rows.empty and not observed_players.empty:
                ball_xy = ball_rows.iloc[0][["LOCATION_X", "LOCATION_Y"]].to_numpy(dtype=float)
                distances = np.hypot(
                    observed_players["LOCATION_X"].to_numpy(dtype=float) - ball_xy[0],
                    observed_players["LOCATION_Y"].to_numpy(dtype=float) - ball_xy[1],
                )
                observed_distances.extend(distances[np.isfinite(distances)].tolist())

        if observed_distances:
            fallback_radius = max(float(np.quantile(observed_distances, 0.99)) * 1.5, 1e-6)
        else:
            player_rows = game_frame[game_frame["PLAYER_ID"] != -1]
            x_span = player_rows["LOCATION_X"].quantile(0.95) - player_rows["LOCATION_X"].quantile(0.05)
            y_span = player_rows["LOCATION_Y"].quantile(0.95) - player_rows["LOCATION_Y"].quantile(0.05)
            fallback_radius = max(float(np.hypot(x_span, y_span)) * 0.005, 1e-6)

        for timestep, timestep_frame in game_frame.groupby("TIMESTEP", sort=False):
            players = timestep_frame[timestep_frame["PLAYER_ID"] != -1].copy()
            if players.empty:
                continue

            observed_owner = players[players["ON_BALL"].fillna(False).astype(bool)]
            if not observed_owner.empty:
                ball = timestep_frame[timestep_frame["PLAYER_ID"] == -1]
                if not ball.empty:
                    ball_xy = ball.iloc[0][["LOCATION_X", "LOCATION_Y"]].to_numpy()
                    distances = observed_owner[["LOCATION_X", "LOCATION_Y"]].sub(ball_xy, axis=1).pow(2).sum(axis=1)
                    owner = observed_owner.loc[distances.idxmin()].copy()
                else:
                    owner = observed_owner.iloc[0].copy()
                source = "on_ball"
                confidence = "high"
            else:
                ball = timestep_frame[timestep_frame["PLAYER_ID"] == -1]
                if ball.empty:
                    continue
                ball_xy = ball.iloc[0][["LOCATION_X", "LOCATION_Y"]].to_numpy()
                distances = players[["LOCATION_X", "LOCATION_Y"]].sub(ball_xy, axis=1).pow(2).sum(axis=1)
                owner = players.loc[distances.idxmin()].copy()
                if float(np.sqrt(distances.loc[owner.name])) > fallback_radius:
                    continue
                source = "nearest_to_ball"
                confidence = "low"

            owner_index = owner.name
            work.loc[owner_index, "owner_player_id"] = int(owner["PLAYER_ID"])
            work.loc[owner_index, "owner_team_id"] = int(owner["TEAM_ID"])
            work.loc[owner_index, "ownership_source"] = source
            work.loc[owner_index, "ownership_confidence"] = confidence

            ball_indices = timestep_frame.index[timestep_frame["PLAYER_ID"] == -1]
            if len(ball_indices):
                work.loc[ball_indices, "owner_player_id"] = int(owner["PLAYER_ID"])
                work.loc[ball_indices, "owner_team_id"] = int(owner["TEAM_ID"])
                work.loc[ball_indices, "ownership_source"] = source
                work.loc[ball_indices, "ownership_confidence"] = confidence

    work["ownership_team_name"] = work["owner_team_id"].map({0: "team_0", 1: "team_1"})
    return work


def build_possession_chains(frame: pd.DataFrame) -> pd.DataFrame:
    """Create simple possession-chain groups by consecutive team owners."""
    if "owner_team_id" not in frame.columns:
        raise ValueError("owner_team_id is required to build possession chains")

    chains = []
    for game_id, game_frame in frame.groupby("GAME_ID", sort=False):
        if "PLAYER_ID" in game_frame.columns and (game_frame["PLAYER_ID"] == -1).any():
            game_frame = game_frame[game_frame["PLAYER_ID"] == -1]
        game_frame = game_frame.sort_values("TIMESTEP").drop_duplicates(subset=["TIMESTEP"], keep="first")
        prior_team = None
        start_frame = None
        chain_players = []
        for _, row in game_frame.iterrows():
            current_team = row["owner_team_id"]
            if pd.isna(current_team):
                if prior_team is not None:
                    chains.append({
                        "GAME_ID": game_id,
                        "team_id": int(prior_team),
                        "start_timestep": int(start_frame),
                        "end_timestep": int(row["TIMESTEP"]),
                        "duration_frames": int(row["TIMESTEP"] - start_frame),
                        "participants": sorted(set(chain_players)),
                    })
                    prior_team = None
                    start_frame = None
                    chain_players = []
                continue
            if prior_team is None:
                prior_team = current_team
                start_frame = int(row["TIMESTEP"])
                chain_players = []
            elif current_team != prior_team:
                chains.append({
                    "GAME_ID": game_id,
                    "team_id": int(prior_team),
                    "start_timestep": int(start_frame),
                    "end_timestep": int(row["TIMESTEP"] - 1 if row["TIMESTEP"] > 0 else row["TIMESTEP"]),
                    "duration_frames": int(row["TIMESTEP"] - start_frame),
                    "participants": sorted(set(chain_players)),
                })
                prior_team = current_team
                start_frame = int(row["TIMESTEP"])
                chain_players = []
            chain_players.append(int(row["owner_player_id"])) if pd.notna(row["owner_player_id"]) else None
        if prior_team is not None:
            chains.append({
                "GAME_ID": game_id,
                "team_id": int(prior_team),
                "start_timestep": int(start_frame),
                "end_timestep": int(game_frame["TIMESTEP"].max()),
                "duration_frames": int(game_frame["TIMESTEP"].max() - start_frame + 1),
                "participants": sorted(set(chain_players)),
            })

    return pd.DataFrame(chains)
