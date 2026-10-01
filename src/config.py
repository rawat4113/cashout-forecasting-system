"""Central configuration. Change values here, nothing else needs editing."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
PLOT_DIR = OUT_DIR / "plots"

SEED = 42

# ---- synthetic data generator -------------------------------------------
START_DATE = "2024-01-01"
N_DAYS = 540                      # ~18 months of history
TARGET_EVENTS_PER_DAY = 45        # average cash-out events/day across all cells (scaled-down India)
UNTRACED_FRACTION = 0.15          # share of cash-outs never traced back to a complaint
MEAN_COMPLAINT_DELAY_H = 20.0     # hours between withdrawal and the complaint reaching the portal

# ---- modelling -----------------------------------------------------------
WARMUP_DAYS = 28                  # first days are dropped (rolling windows not full yet)
TRAIN_FRAC = 0.70                 # chronological split: 70% train / 10% val / 20% test
VAL_FRAC = 0.10
KNN_NEIGHBOURS = 5                # geographic neighbours used for spill-over features

# ---- evaluation / alerting ------------------------------------------------
TOP_K_LIST = (5, 10, 20)          # number of cells an agency can realistically cover per day
ALERT_HOUR = 0                    # hour of day at which the daily forecast is issued
RISK_HIGH = 0.60                  # probability thresholds for alert levels
RISK_MEDIUM = 0.35
