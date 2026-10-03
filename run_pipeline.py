"""One command to reproduce everything:  python run_pipeline.py   (add --regen to rebuild the synthetic data)"""
import sys

from src import config as C
from src import evaluate, export_model, generate_data, predict, train

if __name__ == "__main__":
    if "--regen" in sys.argv or not (C.DATA_DIR / "events.csv").exists():
        print(">> 1/5 generating synthetic data")
        generate_data.main()
    else:
        print(">> 1/5 using existing data in data/ (pass --regen to rebuild)")
    print("\n>> 2/5 training models")
    train.main()
    print("\n>> 3/5 evaluating on the held-out test period")
    evaluate.main()
    print("\n>> 4/5 exporting the portable (numpy-only, endian-explicit) model for IBM Z / LinuxONE")
    export_model.main()
    print("\n>> 5/5 forecasting the next day")
    sys.argv = [sys.argv[0]]
    predict.main()
