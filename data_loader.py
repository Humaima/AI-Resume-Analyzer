import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "job_title_des.csv"


def load_dataset():
    return pd.read_csv(DATA_PATH)
