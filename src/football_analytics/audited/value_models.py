"""Model-readiness registry and interfaces for contextual football value metrics."""
from __future__ import annotations

import pandas as pd

MODEL_INFO = {
    "xG": ("implementation_available_requires_labelled_shots", "A trainable XGBoost xG model is implemented. Fit it with independent shot outcome labels; the tracking-only example has no usable shot outcome labels."),
    "post_shot_xG": ("unavailable", "No dependable on-target shot trajectory/placement labels."),
    "xPass": ("unavailable", "Candidate pass labels are heuristic and no labelled training/calibration set is supplied."),
    "xT": ("interface_ready_values_required", "Zone-to-threat values can be supplied; default zone values are not fabricated."),
    "EPV": ("interface_only", "Requires trained labels or calibrated action/possession-value parameters."),
    "VAEP": ("unavailable", "Requires reliable action labels and scoring/conceding probability models."),
    "pitch_control": ("configurable_baseline", "A native-coordinate constant-speed reachability function is implemented; configure max speed and bounds."),
    "interceptability": ("interface_only", "Pass intent, receiver and calibrated ball trajectory are not generally observed."),
    "pressure_value": ("proxy_only", "Distances and geometry are measurable; pressure/space value is not trained or calibrated."),
}


def build_model_status_table(config: dict) -> pd.DataFrame:
    rows = []
    pitch_cfg = config.get("pitch_control", {})
    pitch_bounds = config.get("pitch", {}).get("bounds_native")
    for name, (status, reason) in MODEL_INFO.items():
        model_status = status
        if name == "pitch_control" and pitch_cfg.get("max_speed_native_per_frame") is not None and pitch_bounds is not None:
            model_status = "configured_uncalibrated_baseline_available"
        rows.append({
            "model_name": name, "model_version": "baseline-0.1",
            "status": model_status, "output_available": False,
            "calibration_status": "not_calibrated",
            "required_training_data": "independent labelled shot outcomes or calibrated model parameters where applicable",
            "assumptions": reason,
            "uncertainty": "Not quantified unless a model with validated uncertainty estimates is supplied.",
            "distinction_from_observed_fact": "All output is inferred/model-derived; it is not a directly observed event.",
        })
    return pd.DataFrame(rows)


def evaluate_xt_delta(start_zone_value: float, end_zone_value: float) -> float:
    """Compute xT delta once calibrated zone values have been supplied externally."""
    return float(end_zone_value) - float(start_zone_value)
