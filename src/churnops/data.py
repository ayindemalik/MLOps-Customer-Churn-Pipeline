"""Pipeline stages 1 and 2: download the raw file, then clean, check and split it."""

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churnops.config import PROCESSED, RAW_CSV, TARGET
from churnops.schema import CLEAN_SCHEMA


def download(url: str, dest: Path = RAW_CSV) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    # 11 brand-new customers (tenure 0) have a blank TotalCharges. They have not
    # been billed yet, so 0 is the true value, not a missing one.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    df = df.drop_duplicates(subset="customerID")
    return CLEAN_SCHEMA.validate(df)


def split(df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    """Train / validation / test, each with the same share of churners."""
    rest, test = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)
    val_share = val_size / (1 - test_size)
    train, val = train_test_split(rest, test_size=val_share, stratify=rest[TARGET], random_state=seed)
    return train, val, test


def prepare(params: dict) -> dict[str, int]:
    df = clean(pd.read_csv(RAW_CSV))
    s = params["split"]
    parts = split(df, s["test_size"], s["val_size"], s["seed"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, part in zip(["train", "val", "test"], parts):
        part.to_parquet(PROCESSED / f"{name}.parquet", index=False)
        sizes[name] = len(part)
    return sizes
