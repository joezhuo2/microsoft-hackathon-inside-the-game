from __future__ import annotations

from typing import Any


def model_status(name: str) -> dict[str, Any]:
    return {
        "model_name": name,
        "version": "baseline-v0",
        "status": "heuristic_proxy",
        "calibration_status": "not_calibrated",
        "training_data_required": "labelled actions or shots",
        "assumptions": "transparent heuristic without trained coefficients",
    }


def xg_baseline(shot_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    payload = model_status("xG_baseline")
    payload["output"] = "unavailable_without_shot_geometry"
    return payload


def xpass_baseline(pass_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    payload = model_status("xPass_baseline")
    payload["output"] = "heuristic_completion_proxy"
    return payload


def xt_baseline(zone_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    payload = model_status("xT_baseline")
    payload["output"] = "illustrative_pitch_zone_proxy"
    return payload
