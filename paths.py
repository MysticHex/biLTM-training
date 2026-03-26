"""
Shared artifact paths for AttnRetrofit project.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

PROCESSED_DIR = ARTIFACTS_DIR / "processed"
MODELS_DIR = ARTIFACTS_DIR / "models"
CONFIG_DIR = ARTIFACTS_DIR / "config"
STUDIES_DIR = ARTIFACTS_DIR / "studies"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
REPORTS_DIR = ARTIFACTS_DIR / "reports"


def ensure_artifact_dirs() -> None:
    """Create artifact directories if they do not exist."""
    for directory in [
        ARTIFACTS_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        CONFIG_DIR,
        STUDIES_DIR,
        METRICS_DIR,
        PLOTS_DIR,
        REPORTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
