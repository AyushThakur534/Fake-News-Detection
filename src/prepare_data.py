"""Prepare Fake.csv and True.csv into data/news.csv."""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def normalize_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()

def read_source(path, label):
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in ("title", "text", "subject", "date"):
        if col not in df.columns:
            df[col] = ""
    df["label"] = label
    return df[["title", "text", "subject", "date", "label"]].copy()

def main():
    print("Reading original CSV files...")
    df = pd.concat([
        read_source(DATA / "Fake.csv", "FAKE"),
        read_source(DATA / "True.csv", "REAL")
    ], ignore_index=True)
    print(f"Rows loaded: {len(df):,}")
    print("\nMissing values before cleaning:\n", df.isna().sum())

    for col in ["title", "text", "subject", "date"]:
        df[col] = df[col].map(normalize_text)
    df = df[df["text"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["title", "text", "label"])

    df["_key"] = (
        df["title"].str.lower().str.strip() + " " +
        df["text"].str.lower().str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    counts = df.groupby("_key")["label"].nunique()
    conflicting = counts[counts > 1].index
    print(f"Conflicting article groups removed: {len(conflicting):,}")
    if len(conflicting):
        df = df[~df["_key"].isin(conflicting)].copy()
    df = df.drop(columns=["_key"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    output = DATA / "news.csv"
    df.to_csv(output, index=False)
    print(f"\nSaved: {output}")
    print(f"Final rows: {len(df):,}")
    print("\nClass distribution:\n", df["label"].value_counts())
    print("\nDate examples:\n", df["date"].head().to_string(index=False))
    print("\nsubject/date are retained for analysis only, not model features.")

if __name__ == "__main__":
    main()
