"""Football tracking analytics package."""

from .config import AnalyticsConfig
from .data_loader import load_tracking_data
from .preprocessing import validate_tracking_frame
from .tracking_features import compute_tracking_features
from .possession import infer_possession
from .event_detection import detect_events
from .quality_checks import generate_quality_report

__version__ = "0.1.0"
