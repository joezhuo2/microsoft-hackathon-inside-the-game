from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PARAMETERS: dict[str, Any] = {
    "control_state_machine": {
        "acquisition_radius_m": 1.25,
        "retention_radius_m": 2.25,
        "contested_radius_m": 2.5,
        "contested_player_distance_difference_max_m": 0.75,
        "stable_control_min_consecutive_frames": 3,
    },
    "primitive_event_detection": {
        "touch_candidate": {
            "player_ball_distance_max_m": 1.25,
            "supporting_ball_delta_velocity_min_m_per_timestep": 0.5,
            "supporting_turn_angle_min_degrees": 45.0,
            "turn_angle_requires_speed_on_both_sides_above_m_per_timestep": 0.75,
        },
        "ball_release_candidate": {
            "ball_speed_support_threshold_m_per_timestep": 1.0,
            "controller_separation_exceeds_retention_radius_m": 2.25,
            "separation_deadline_frames": 3,
        },
        "event_linking": {"maximum_unresolved_event_window_frames": 30},
    },
    "movement_events": {
        "carry": {"min_ball_path_length_m": 1.0},
        "progressive_carry": {"net_goalward_displacement_min_m": 10.0},
    },
    "shot_and_defensive_events": {
        "shot_candidate": {"goal_half_width_m": 3.66, "goal_plane_margin_m": 1.0},
    },
    "out_of_play_and_restarts": {
        "boundary_detection": {"outside_margin_m": 0.25},
    },
    "attacking_direction": {"initial_sample_inference": {"team_0": 1, "team_1": -1}},
}


def _nested(parameters: Mapping[str, Any], *keys: str) -> Any:
    value: Any = parameters
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _event(
    game_id: Any,
    timestep: int,
    event_type: str,
    *,
    end_timestep: int | None = None,
    subtype: str | None = None,
    player_id: Any = None,
    team_id: Any = None,
    previous_controller: Any = None,
    next_controller: Any = None,
    start_position: tuple[float, float] | None = None,
    end_position: tuple[float, float] | None = None,
    confidence: str = "LOW",
    evidence: tuple[str, ...] = (),
    alternatives: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "GAME_ID": game_id,
        "start_timestep": timestep,
        "end_timestep": timestep if end_timestep is None else end_timestep,
        "event_type": event_type,
        "subtype": subtype,
        "PLAYER_ID": player_id,
        "TEAM_ID": team_id,
        "previous_controller": previous_controller,
        "next_controller": next_controller,
        "start_x": start_position[0] if start_position else np.nan,
        "start_y": start_position[1] if start_position else np.nan,
        "end_x": end_position[0] if end_position else np.nan,
        "end_y": end_position[1] if end_position else np.nan,
        "confidence_level": confidence,
        "evidence_flags": ";".join(evidence),
        "alternative_hypotheses": ";".join(alternatives),
        "rule_version": "tracking-rules-v1",
    }


def _control_states(
    frames: list[dict[str, Any]], parameters: Mapping[str, Any]
) -> list[dict[str, Any]]:
    settings = _nested(parameters, "control_state_machine") or {}
    acquire = float(settings.get("acquisition_radius_m", 1.25))
    retain = float(settings.get("retention_radius_m", 2.25))
    contested_radius = float(settings.get("contested_radius_m", 2.5))
    tie_limit = float(settings.get("contested_player_distance_difference_max_m", 0.75))
    persistence = int(settings.get("stable_control_min_consecutive_frames", 3))

    candidates: list[Any] = []
    states: list[str] = []
    for frame in frames:
        players = frame["players"]
        distances = frame["distances"]
        if len(players) == 0 or len(distances) == 0:
            candidates.append(None)
            states.append("UNKNOWN")
            continue
        order = np.argsort(distances)
        nearest = int(order[0])
        nearest_distance = float(distances[nearest])
        if nearest_distance > contested_radius:
            candidates.append(None)
            states.append("FREE_BALL")
        elif len(order) > 1 and float(distances[int(order[1])] - distances[nearest]) <= tie_limit and float(distances[int(order[1])]) <= contested_radius:
            candidates.append(None)
            states.append("CONTESTED")
        elif nearest_distance <= acquire:
            candidates.append(int(players.iloc[nearest]["PLAYER_ID"]))
            states.append("CONTROL_CANDIDATE")
        else:
            candidates.append(None)
            states.append("FREE_BALL")

    stable: list[Any] = [None] * len(frames)
    index = 0
    while index < len(frames):
        candidate = candidates[index]
        stop = index + 1
        while stop < len(frames) and candidates[stop] == candidate and frames[stop]["timestep"] == frames[stop - 1]["timestep"] + 1:
            stop += 1
        if candidate is not None and stop - index >= persistence:
            for stable_index in range(index, stop):
                stable[stable_index] = candidate
        index = stop

    prior_controller = None
    outside_count = 0
    for index, frame in enumerate(frames):
        if states[index] == "CONTESTED" or frame.get("uncertain_transition", False):
            stable[index] = None
            prior_controller = None
            outside_count = 0
            frame["controller"] = None
            frame["state"] = "CONTESTED" if states[index] == "CONTESTED" else "UNKNOWN"
            continue
        controller = stable[index]
        if controller is not None:
            prior_controller = controller
            outside_count = 0
        elif prior_controller is not None:
            player_match = frame["players"][frame["players"]["PLAYER_ID"] == prior_controller]
            if len(player_match) and np.hypot(
                float(player_match.iloc[0]["LOCATION_X"]) - frame["ball_x"],
                float(player_match.iloc[0]["LOCATION_Y"]) - frame["ball_y"],
            ) <= retain:
                stable[index] = prior_controller
                outside_count = 0
            else:
                outside_count += 1
                if outside_count <= 1:
                    stable[index] = prior_controller
                else:
                    prior_controller = None
        frame["controller"] = stable[index]
        if frame["state"] != "CONTESTED" and stable[index] is not None:
            frame["state"] = "CONTROLLED"
        elif frame["state"] != "CONTESTED" and frame["state"] == "CONTROL_CANDIDATE":
            frame["state"] = "UNKNOWN"
        else:
            frame["state"] = states[index]
    return frames


def _make_game_frames(game_frame: pd.DataFrame) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for timestep, rows in game_frame.groupby("TIMESTEP", sort=True):
        ball_rows = rows[rows["PLAYER_ID"] == -1]
        players = rows[rows["PLAYER_ID"] != -1].copy()
        if ball_rows.empty:
            frames.append({"timestep": int(timestep), "players": players, "distances": np.array([]), "ball_x": np.nan, "ball_y": np.nan, "ball_vx": np.nan, "ball_vy": np.nan, "state": "UNKNOWN"})
            continue
        ball = ball_rows.iloc[0]
        distances = np.hypot(players["LOCATION_X"].to_numpy(dtype=float) - float(ball["LOCATION_X"]), players["LOCATION_Y"].to_numpy(dtype=float) - float(ball["LOCATION_Y"]))
        frames.append({
            "timestep": int(timestep),
            "players": players,
            "distances": distances,
            "ball_x": float(ball["LOCATION_X"]),
            "ball_y": float(ball["LOCATION_Y"]),
            "ball_vx": float(ball["VELOCITY_X"]) if pd.notna(ball["VELOCITY_X"]) else 0.0,
            "ball_vy": float(ball["VELOCITY_Y"]) if pd.notna(ball["VELOCITY_Y"]) else 0.0,
            "state": "UNKNOWN",
        })
    for index in range(1, len(frames)):
        before, after = frames[index - 1], frames[index]
        continuous = after["timestep"] == before["timestep"] + 1
        ball_jump = float(np.hypot(after["ball_x"] - before["ball_x"], after["ball_y"] - before["ball_y"])) if continuous else 0.0
        previous_players = before["players"].set_index("PLAYER_ID")
        current_players = after["players"].set_index("PLAYER_ID")
        shared_ids = previous_players.index.intersection(current_players.index)
        player_jumps = 0
        if continuous and len(shared_ids):
            player_dx = current_players.loc[shared_ids, "LOCATION_X"].to_numpy(dtype=float) - previous_players.loc[shared_ids, "LOCATION_X"].to_numpy(dtype=float)
            player_dy = current_players.loc[shared_ids, "LOCATION_Y"].to_numpy(dtype=float) - previous_players.loc[shared_ids, "LOCATION_Y"].to_numpy(dtype=float)
            player_jumps = int(np.count_nonzero(np.hypot(player_dx, player_dy) > 3.0))
        after["uncertain_transition"] = not continuous or ball_jump > 10.0 or player_jumps >= 8
    return frames


def _intervals(frames: list[dict[str, Any]]) -> list[tuple[int, int, Any]]:
    intervals: list[tuple[int, int, Any]] = []
    start = 0
    while start < len(frames):
        controller = frames[start]["controller"]
        if controller is None:
            start += 1
            continue
        end = start + 1
        while end < len(frames) and frames[end]["controller"] == controller and frames[end]["timestep"] == frames[end - 1]["timestep"] + 1:
            end += 1
        intervals.append((start, end - 1, controller))
        start = end
    return intervals


def _detect_game(game_id: Any, game_frame: pd.DataFrame, parameters: Mapping[str, Any]) -> list[dict[str, Any]]:
    frames = _control_states(_make_game_frames(game_frame), parameters)
    events: list[dict[str, Any]] = []
    intervals = _intervals(frames)
    player_teams = game_frame[game_frame["PLAYER_ID"] != -1].drop_duplicates("PLAYER_ID").set_index("PLAYER_ID")["TEAM_ID"].to_dict()
    attack = _nested(parameters, "attacking_direction", "initial_sample_inference") or {"team_0": 1, "team_1": -1}

    previous_velocity: tuple[float, float] | None = None
    previous_timestep: int | None = None
    touch_settings = _nested(parameters, "primitive_event_detection", "touch_candidate") or {}
    touch_radius = float(touch_settings.get("player_ball_distance_max_m", 1.25))
    delta_v_min = float(touch_settings.get("supporting_ball_delta_velocity_min_m_per_timestep", 0.5))
    turn_min = float(touch_settings.get("supporting_turn_angle_min_degrees", 45.0))
    for frame in frames:
        velocity = (frame["ball_vx"], frame["ball_vy"])
        if previous_velocity is not None and previous_timestep is not None and frame["timestep"] == previous_timestep + 1 and not frame.get("uncertain_transition", False):
            delta_v = float(np.hypot(velocity[0] - previous_velocity[0], velocity[1] - previous_velocity[1]))
            old_speed, new_speed = float(np.hypot(*previous_velocity)), float(np.hypot(*velocity))
            cosine = np.dot(previous_velocity, velocity) / max(old_speed * new_speed, 1e-9)
            turn = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
            if delta_v >= delta_v_min or (old_speed >= 0.75 and new_speed >= 0.75 and turn >= turn_min):
                if len(frame["distances"]):
                    nearest_index = int(np.argmin(frame["distances"]))
                    if float(frame["distances"][nearest_index]) <= touch_radius:
                        player = frame["players"].iloc[nearest_index]
                        events.append(_event(game_id, frame["timestep"], "touch_candidate", player_id=int(player["PLAYER_ID"]), team_id=int(player["TEAM_ID"]), start_position=(frame["ball_x"], frame["ball_y"]), confidence="LOW", evidence=("close_proximity", "ball_trajectory_change"), alternatives=("deflection", "tracking_noise")))
        previous_velocity = velocity
        previous_timestep = frame["timestep"]

    carry_settings = _nested(parameters, "movement_events", "carry") or {}
    carry_min = float(carry_settings.get("min_ball_path_length_m", 1.0))
    progressive_min = float(_nested(parameters, "movement_events", "progressive_carry", "net_goalward_displacement_min_m") or 10.0)
    for start, end, controller in intervals:
        if end <= start:
            continue
        player_id = int(controller)
        team_id = player_teams.get(player_id)
        team_frames = frames[start:end + 1]
        path = sum(float(np.hypot(right["ball_x"] - left["ball_x"], right["ball_y"] - left["ball_y"])) for left, right in zip(team_frames, team_frames[1:]) if right["timestep"] == left["timestep"] + 1)
        first, last = team_frames[0], team_frames[-1]
        if path >= carry_min:
            evidence = ["stable_control", "ball_path_progression"]
            subtype = None
            if team_id in (0, 1):
                direction = float(attack.get(f"team_{team_id}", 0.0))
                if direction and (last["ball_x"] - first["ball_x"]) * direction >= progressive_min:
                    subtype = "progressive"
                    evidence.append("goalward_displacement")
            events.append(_event(game_id, first["timestep"], "carry", end_timestep=last["timestep"], subtype=subtype, player_id=player_id, team_id=team_id, start_position=(first["ball_x"], first["ball_y"]), end_position=(last["ball_x"], last["ball_y"]), confidence="MEDIUM", evidence=tuple(evidence)))
            if team_id in (0, 1):
                start_opponents = team_frames[0]["players"][team_frames[0]["players"]["TEAM_ID"] != team_id]
                end_opponents = team_frames[-1]["players"][team_frames[-1]["players"]["TEAM_ID"] != team_id]
                if not start_opponents.empty and not end_opponents.empty:
                    initial_pressure = float(np.hypot(start_opponents["LOCATION_X"] - first["ball_x"], start_opponents["LOCATION_Y"] - first["ball_y"]).min())
                    final_pressure = float(np.hypot(end_opponents["LOCATION_X"] - last["ball_x"], end_opponents["LOCATION_Y"] - last["ball_y"]).min())
                    if initial_pressure <= 2.5 and final_pressure >= initial_pressure + 1.0:
                        events.append(_event(game_id, first["timestep"], "dribble_candidate", end_timestep=last["timestep"], player_id=player_id, team_id=team_id, start_position=(first["ball_x"], first["ball_y"]), end_position=(last["ball_x"], last["ball_y"]), confidence="LOW", evidence=("sustained_carry", "nearby_opponent_bypassed"), alternatives=("defender_repositioning", "tracking_uncertainty")))
        if start == 0 or frames[start - 1]["controller"] != controller:
            prior_controller = frames[start - 1]["controller"] if start else None
            prior_team = player_teams.get(int(prior_controller)) if prior_controller is not None else None
            if prior_controller is None or prior_team != team_id:
                events.append(_event(game_id, first["timestep"], "possession_gain_candidate", player_id=player_id, team_id=team_id, previous_controller=prior_controller, next_controller=player_id, start_position=(first["ball_x"], first["ball_y"]), confidence="MEDIUM", evidence=("persistent_control", "prior_free_or_other_team_control")))

    release_settings = _nested(parameters, "primitive_event_detection", "ball_release_candidate") or {}
    release_speed_min = float(release_settings.get("ball_speed_support_threshold_m_per_timestep", 1.0))
    separation_min = float(release_settings.get("controller_separation_exceeds_retention_radius_m", 2.25))
    deadline = int(release_settings.get("separation_deadline_frames", 3))
    resolution_window = int(_nested(parameters, "primitive_event_detection", "event_linking", "maximum_unresolved_event_window_frames") or 30)
    shot_settings = _nested(parameters, "shot_and_defensive_events", "shot_candidate") or {}
    goal_half_width = float(shot_settings.get("goal_half_width_m", 3.66))
    goal_margin = float(shot_settings.get("goal_plane_margin_m", 1.0))

    for start, end, controller in intervals:
        release_index = None
        for candidate_index in range(end + 1, min(end + deadline + 1, len(frames))):
            candidate = frames[candidate_index]
            if candidate.get("uncertain_transition", False):
                break
            player = candidate["players"][candidate["players"]["PLAYER_ID"] == controller]
            if player.empty:
                continue
            separation = float(np.hypot(float(player.iloc[0]["LOCATION_X"]) - candidate["ball_x"], float(player.iloc[0]["LOCATION_Y"]) - candidate["ball_y"]))
            speed = float(np.hypot(candidate["ball_vx"], candidate["ball_vy"]))
            if separation >= separation_min and speed >= release_speed_min:
                release_index = candidate_index
                break
        if release_index is None:
            continue

        release = frames[release_index]
        source_team = player_teams.get(int(controller))
        resolution_end = min(release_index + resolution_window, len(frames) - 1)
        next_interval = next((interval for interval in intervals if interval[0] > release_index and interval[0] <= resolution_end), None)
        next_start, _, next_controller = next_interval if next_interval else (None, None, None)
        next_team = player_teams.get(int(next_controller)) if next_controller is not None else None
        destination_index = next_start if next_start is not None else resolution_end
        destination = frames[destination_index]
        dx, dy = destination["ball_x"] - release["ball_x"], destination["ball_y"] - release["ball_y"]
        speed = float(np.hypot(release["ball_vx"], release["ball_vy"]))
        is_shot = False
        shot_goal_x = None
        if source_team in (0, 1) and abs(release["ball_vx"]) > 1e-9:
            direction = float(attack.get(f"team_{source_team}", 0.0))
            target_x = 120.0 if direction > 0 else 0.0
            moving_goalward = release["ball_vx"] * direction > 0
            if moving_goalward:
                time_to_goal = (target_x - release["ball_x"]) / release["ball_vx"]
                projected_y = release["ball_y"] + release["ball_vy"] * time_to_goal
                is_shot = time_to_goal >= 0 and abs(projected_y - 40.0) <= goal_half_width + goal_margin
                shot_goal_x = target_x

        event_type = "shot_candidate" if is_shot else "pass_attempt" if next_team == source_team else "incomplete_pass_candidate" if next_team is not None and next_team != source_team else "ball_release_candidate"
        subtype = "planar_on_target_candidate" if is_shot else None
        outcome = "completed_pass_candidate" if event_type == "pass_attempt" else "interception_candidate" if event_type == "incomplete_pass_candidate" else "unresolved" if next_controller is None else None
        evidence = ["stable_prior_control", "controller_ball_separation", "independent_ball_motion"]
        alternatives = ["long_pass"] if is_shot else []
        if next_team == source_team:
            evidence.append("teammate_control_after_release")
        elif next_team is not None and next_team != source_team:
            evidence.append("opponent_control_after_release")
        if speed > 0 and abs(dx) + abs(dy) > 0:
            evidence.append("outgoing_trajectory")
        if event_type in {"pass_attempt", "incomplete_pass_candidate"} and next_start is not None:
            direction = float(attack.get(f"team_{source_team}", 0.0)) if source_team in (0, 1) else 0.0
            origin_wide = min(release["ball_y"], 80.0 - release["ball_y"]) <= 16.0
            target_penalty_x = 120.0 if direction > 0 else 0.0
            enters_penalty = abs(frames[next_start]["ball_x"] - target_penalty_x) <= 16.5 and 19.84 <= frames[next_start]["ball_y"] <= 60.16
            start_wide = release["ball_y"] <= 20.0 or release["ball_y"] >= 60.0
            end_wide = frames[next_start]["ball_y"] <= 20.0 or frames[next_start]["ball_y"] >= 60.0
            if origin_wide and enters_penalty:
                subtype = "cross_candidate"
                evidence.append("wide_origin_and_penalty_area_trajectory")
            elif start_wide != end_wide and abs(frames[next_start]["ball_y"] - release["ball_y"]) >= 30.0:
                subtype = "switch_of_play_candidate"
                evidence.append("opposite_wide_channels")
            else:
                opponents = frames[start]["players"][frames[start]["players"]["TEAM_ID"] != source_team]
                receiver_is_behind_line = False
                if direction and not opponents.empty:
                    defensive_line = opponents["LOCATION_X"].quantile(0.25 if direction > 0 else 0.75)
                    receiver_is_behind_line = (frames[next_start]["ball_x"] - defensive_line) * direction >= 3.0
                if receiver_is_behind_line:
                    subtype = "through_ball_candidate"
                    evidence.append("receiver_beyond_robust_defensive_line")
        record = _event(game_id, release["timestep"], event_type, end_timestep=frames[destination_index]["timestep"], subtype=subtype, player_id=int(controller), team_id=source_team, previous_controller=int(controller), next_controller=next_controller, start_position=(release["ball_x"], release["ball_y"]), end_position=(destination["ball_x"], destination["ball_y"]), confidence="MEDIUM" if event_type != "ball_release_candidate" or is_shot else "UNRESOLVED", evidence=tuple(evidence), alternatives=tuple(alternatives))
        if outcome:
            record["outcome"] = outcome
        events.append(record)
        if next_controller is not None:
            received = frames[next_start]
            receiver_type = "reception_candidate" if next_team == source_team else "interception_candidate"
            events.append(_event(game_id, received["timestep"], receiver_type, player_id=int(next_controller), team_id=next_team, previous_controller=int(controller), next_controller=int(next_controller), start_position=(received["ball_x"], received["ball_y"]), confidence="MEDIUM", evidence=("release_linked", "stable_new_control")))
            if next_team != source_team:
                events.append(_event(game_id, release["timestep"], "possession_loss_candidate", player_id=int(controller), team_id=source_team, previous_controller=int(controller), next_controller=int(next_controller), start_position=(release["ball_x"], release["ball_y"]), end_position=(received["ball_x"], received["ball_y"]), confidence="MEDIUM", evidence=("opponent_stable_control", "release_linked")))

        if not is_shot and source_team in (0, 1):
            direction = float(attack.get(f"team_{source_team}", 0.0))
            own_goal_distance = release["ball_x"] if direction > 0 else 120.0 - release["ball_x"]
            away_from_own_goal = (dx * direction) > 0
            if own_goal_distance <= 40.0 and away_from_own_goal and dx * direction >= 10.0 and speed >= release_speed_min:
                events.append(_event(game_id, release["timestep"], "clearance_candidate", player_id=int(controller), team_id=source_team, previous_controller=int(controller), start_position=(release["ball_x"], release["ball_y"]), end_position=(destination["ball_x"], destination["ball_y"]), confidence="LOW", evidence=("defensive_region_release", "long_ball_away_from_own_goal", "intent_not_observable"), alternatives=("long_pass", "failed_pass")))

        if is_shot and shot_goal_x is not None:
            for index in range(release_index + 1, min(release_index + resolution_window + 1, len(frames))):
                before, after = frames[index - 1], frames[index]
                if after.get("uncertain_transition", False):
                    break
                crossed = (before["ball_x"] - shot_goal_x) * (after["ball_x"] - shot_goal_x) <= 0 and before["timestep"] + 1 == after["timestep"]
                if crossed and abs(after["ball_y"] - 40.0) <= goal_half_width:
                    events.append(_event(game_id, after["timestep"], "goal_candidate", player_id=int(controller), team_id=source_team, previous_controller=int(controller), start_position=(before["ball_x"], before["ball_y"]), end_position=(after["ball_x"], after["ball_y"]), confidence="LOW", evidence=("goal_plane_crossing", "goal_mouth_y"), alternatives=("wide_or_blocked_shot",)))
                    break

    boundary_margin = float(_nested(parameters, "out_of_play_and_restarts", "boundary_detection", "outside_margin_m") or 0.25)
    boundary_indices: list[int] = []
    for index in range(1, len(frames)):
        before, after = frames[index - 1], frames[index]
        if after["timestep"] != before["timestep"] + 1 or after.get("uncertain_transition", False):
            continue
        if not np.isfinite([after["ball_x"], after["ball_y"]]).all():
            continue
        was_inside = 0.0 <= before["ball_x"] <= 120.0 and 0.0 <= before["ball_y"] <= 80.0
        is_outside = after["ball_x"] < -boundary_margin or after["ball_x"] > 120.0 + boundary_margin or after["ball_y"] < -boundary_margin or after["ball_y"] > 80.0 + boundary_margin
        if was_inside and is_outside:
            boundary_indices.append(index)
            events.append(_event(game_id, after["timestep"], "boundary_exit_candidate", start_position=(before["ball_x"], before["ball_y"]), end_position=(after["ball_x"], after["ball_y"]), confidence="LOW", evidence=("segment_crosses_pitch_boundary",), alternatives=("tracking_discontinuity",)))

    stationary_min = float(_nested(parameters, "control_state_machine", "stationary_ball_speed_threshold_m_per_timestep") or 0.25)
    stationary_frames = int(_nested(parameters, "control_state_machine", "stationary_ball_min_frames") or 3)
    index = 0
    while index < len(frames):
        speed = float(np.hypot(frames[index]["ball_vx"], frames[index]["ball_vy"]))
        if speed > stationary_min or frames[index]["controller"] is not None:
            index += 1
            continue
        stop = index + 1
        while stop < len(frames) and frames[stop]["timestep"] == frames[stop - 1]["timestep"] + 1 and frames[stop]["controller"] is None and np.hypot(frames[stop]["ball_vx"], frames[stop]["ball_vy"]) <= stationary_min:
            stop += 1
        if stop - index >= stationary_frames:
            if any(boundary_index < stop and boundary_index >= max(0, index - resolution_window) for boundary_index in boundary_indices):
                restart_type = "restart_candidate"
                restart_subtype = "corner_candidate" if (frames[index]["ball_x"] <= 1.5 or frames[index]["ball_x"] >= 118.5) and (frames[index]["ball_y"] <= 1.5 or frames[index]["ball_y"] >= 78.5) else "throw_in_candidate" if frames[index]["ball_y"] <= 1.5 or frames[index]["ball_y"] >= 78.5 else "goal_kick_candidate" if frames[index]["ball_x"] <= 5.5 or frames[index]["ball_x"] >= 114.5 else "other_restart_candidate"
                events.append(_event(game_id, frames[index]["timestep"], restart_type, end_timestep=frames[stop - 1]["timestep"], subtype=restart_subtype, start_position=(frames[index]["ball_x"], frames[index]["ball_y"]), confidence="LOW", evidence=("prior_boundary_exit", "stationary_uncontrolled_ball"), alternatives=("ordinary_ball_stop",)))
            else:
                ball_x, ball_y = frames[index]["ball_x"], frames[index]["ball_y"]
                restart_subtype = None
                evidence = ["stationary_uncontrolled_ball"]
                if np.hypot(ball_x - 60.0, ball_y - 40.0) <= 1.5:
                    team_mean_x = game_frame[game_frame["PLAYER_ID"] != -1].groupby("TEAM_ID")["LOCATION_X"].mean()
                    if len(team_mean_x) >= 2 and team_mean_x.min() < 60.0 < team_mean_x.max():
                        restart_subtype = "kickoff_candidate"
                        evidence.append("ball_at_center_with_opposing_half_formation")
                for goal_x, attack_sign in ((0.0, 1.0), (120.0, -1.0)):
                    spot_x = goal_x + attack_sign * 11.0
                    if np.hypot(ball_x - spot_x, ball_y - 40.0) <= 1.0:
                        nearby_players = frames[index]["players"]
                        keeper_near_goal = not nearby_players.empty and float((nearby_players["LOCATION_X"] - goal_x).abs().min()) <= 3.0
                        players_outside_box = not nearby_players.empty and not ((nearby_players["LOCATION_X"].sub(goal_x).abs() <= 16.5) & (nearby_players["LOCATION_Y"].between(19.84, 60.16))).any()
                        if keeper_near_goal and players_outside_box:
                            restart_subtype = "penalty_candidate"
                            evidence.extend(["ball_near_penalty_spot", "goal_line_player", "other_players_outside_penalty_area"])
                event_type = "restart_candidate" if restart_subtype else "dead_ball_candidate"
                events.append(_event(game_id, frames[index]["timestep"], event_type, end_timestep=frames[stop - 1]["timestep"], subtype=restart_subtype, start_position=(ball_x, ball_y), confidence="LOW" if restart_subtype else "UNRESOLVED", evidence=tuple(evidence), alternatives=("ordinary_ball_stop", "tracking_gap")))
        index = stop

    return events


def detect_events(frame: pd.DataFrame, parameters: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Infer candidate events from positions, velocities, and entity identifiers only."""
    required = {"GAME_ID", "TIMESTEP", "PLAYER_ID", "TEAM_ID", "LOCATION_X", "LOCATION_Y", "VELOCITY_X", "VELOCITY_Y"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing tracking-only event columns: {sorted(missing)}")
    settings = dict(DEFAULT_PARAMETERS)
    if parameters:
        settings.update(parameters)
    events: list[dict[str, Any]] = []
    for game_id, game_frame in frame.groupby("GAME_ID", sort=False):
        events.extend(_detect_game(game_id, game_frame, settings))
    result = pd.DataFrame(events)
    if not result.empty:
        result.insert(0, "event_id", [f"{game_id}-{index:06d}" for index, game_id in enumerate(result["GAME_ID"], start=1)])
        result = result.sort_values(["GAME_ID", "start_timestep", "event_type"], kind="mergesort").reset_index(drop=True)
        result["event_id"] = [f"{row.GAME_ID}-{index:06d}" for index, row in enumerate(result.itertuples(index=False), start=1)]
    return result


def detect_pass_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper returning pass attempts from the tracking-only detector."""
    events = detect_events(frame)
    return events[events["event_type"] == "pass_attempt"].copy() if not events.empty else events