"""Export the trained sklearn model to the portable numpy-only format used on IBM Z / LinuxONE.

    python -m src.export_model
"""
import joblib

from . import config as C
from .features import FEATURE_COLUMNS
from .portable_model import export_hgb


def main():
    model = joblib.load(C.MODEL_DIR / "hgb_model.joblib")
    sha = export_hgb(model, C.PORTABLE_MODEL_PATH, FEATURE_COLUMNS)
    print(f"portable model written: {C.PORTABLE_MODEL_PATH.name}  sha256={sha[:16]}…")


if __name__ == "__main__":
    main()
