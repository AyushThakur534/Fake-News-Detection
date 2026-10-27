"""Train and compare classifiers; includes random and optional chronological holdouts."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from src.preprocess import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LABELS = ["FAKE", "REAL"]

def normalize_label(value):
    s = str(value).strip().lower()
    if s in {"fake", "false", "0", "unreliable"}: return "FAKE"
    if s in {"real", "true", "1", "reliable"}: return "REAL"
    return None

def load_dataset(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "label" not in df or "text" not in df:
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    if "title" not in df: df["title"] = ""
    if "date" not in df: df["date"] = ""
    df["label"] = df["label"].map(normalize_label)
    df["combined_text"] = (
        df["title"].fillna("").astype(str) + " " +
        df["text"].fillna("").astype(str)
    ).map(clean_text)
    df = df.dropna(subset=["label"])
    df = df[df["combined_text"].str.len() > 0].copy()
    df["_key"] = df["combined_text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    counts = df.groupby("_key")["label"].nunique()
    conflicting = counts[counts > 1].index
    if len(conflicting):
        print(f"Removing {len(conflicting):,} duplicate-text groups with conflicting labels.")
        df = df[~df["_key"].isin(conflicting)].copy()
    df = df.drop_duplicates("_key").drop(columns=["_key"])
    if df["label"].nunique() != 2 or df["label"].value_counts().min() < 2:
        raise ValueError("Need at least two examples of both REAL and FAKE.")
    df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return df.reset_index(drop=True)

def classifiers():
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=40, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42),
    }

def pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.98)),
        ("classifier", clf),
    ])

def metrics(y, pred):
    p, r, f, _ = precision_recall_fscore_support(
        y, pred, average="binary", pos_label="FAKE", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "fake_precision": float(p), "fake_recall": float(r), "fake_f1": float(f)
    }

def evaluate(name, train_df, test_df):
    if train_df.label.nunique() < 2 or test_df.label.nunique() < 2:
        print(f"Skipping {name}: both labels are not present in train and test.")
        return None
    rows, fitted = [], {}
    for model_name, clf in classifiers().items():
        print(f"\n{'=' * 55}\n{name}: training {model_name}")
        model = pipeline(clf)
        model.fit(train_df.combined_text, train_df.label)
        pred = model.predict(test_df.combined_text)
        rows.append({"evaluation": name, "model": model_name, **metrics(test_df.label, pred)})
        fitted[model_name] = model
        print(classification_report(test_df.label, pred, labels=LABELS, zero_division=0))
        print("Confusion matrix (actual rows / predicted columns; FAKE, REAL):")
        print(confusion_matrix(test_df.label, pred, labels=LABELS))
    return pd.DataFrame(rows), fitted

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/news.csv")
    ap.add_argument("--test-size", type=float, default=0.20)
    args = ap.parse_args()
    if not 0.1 <= args.test_size <= 0.4:
        raise ValueError("--test-size must be between 0.1 and 0.4.")
    path = Path(args.data)
    if not path.is_absolute(): path = ROOT / path
    if not path.exists(): raise FileNotFoundError(f"Dataset not found: {path}")

    df = load_dataset(path)
    print(f"Usable articles: {len(df):,}\nClass distribution:\n{df.label.value_counts()}")
    train_r, test_r = train_test_split(
        df, test_size=args.test_size, random_state=42, stratify=df.label
    )
    random_result = evaluate("random_holdout", train_r, test_r)
    if random_result is None: raise RuntimeError("Random holdout failed.")
    random_table, random_models = random_result
    result_tables = [random_table]
    selected_table, selected_models, selection_name = random_table, random_models, "random_holdout"
    temporal_available = False

    dated = df[df.parsed_date.notna()].sort_values("parsed_date")
    if len(dated) >= 100:
        split_at = max(1, min(int(len(dated) * (1 - args.test_size)), len(dated) - 1))
        cutoff = dated.iloc[split_at].parsed_date
        old = dated[dated.parsed_date < cutoff]
        new = dated[dated.parsed_date >= cutoff]
        if len(old) >= 50 and len(new) >= 20 and old.label.nunique() == 2 and new.label.nunique() == 2:
            temporal_result = evaluate("chronological_holdout", old, new)
            if temporal_result:
                temporal_table, temporal_models = temporal_result
                result_tables.append(temporal_table)
                selected_table, selected_models, selection_name = temporal_table, temporal_models, "chronological_holdout"
                temporal_available = True
                print(f"Temporal test begins at: {cutoff}")

    all_results = pd.concat(result_tables, ignore_index=True)
    all_results.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
    ranked = selected_table.sort_values(
        ["fake_f1", "balanced_accuracy", "accuracy"], ascending=False
    ).reset_index(drop=True)
    best_name = ranked.iloc[0].model
    joblib.dump(selected_models[best_name], MODEL_DIR / "fake_news_pipeline.joblib")

    summary = {
        "best_model": best_name,
        "selection_evaluation": selection_name,
        "rows_used": int(len(df)),
        "class_distribution": {str(k): int(v) for k, v in df.label.value_counts().items()},
        "random_holdout": {"train_rows": int(len(train_r)), "test_rows": int(len(test_r))},
        "chronological_holdout_available": temporal_available,
        "limitations": [
            "Random split can overestimate performance when source/topic patterns repeat.",
            "Chronological split does not prove generalization to unseen publishers.",
            "Reliable publisher identifiers are not available in this dataset.",
            "Text classification is not factual verification."
        ],
        "selection_rule": "FAKE F1, then balanced accuracy, then accuracy"
    }
    (MODEL_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nMODEL COMPARISON\n", all_results.round(4).to_string(index=False))
    print(f"\nSelected model: {best_name} ({selection_name})")
    print(f"Saved model: {MODEL_DIR / 'fake_news_pipeline.joblib'}")

if __name__ == "__main__":
    main()
