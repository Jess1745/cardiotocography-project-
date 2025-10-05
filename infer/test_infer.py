import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Minimal normalization to tolerate casing/spaces
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    df = df.rename(columns={c: c.lower() for c in df.columns})
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV with required CTG features")
    ap.add_argument("--model", default="models/ctg_best_calibrated.joblib", help="Path to trained joblib")
    ap.add_argument("--out_csv", default="predictions.csv", help="Where to write predictions")
    args = ap.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.csv)
    df = normalize_cols(df)

    X = df.drop(columns=["nsp"], errors="ignore")  # NSP may or may not be present
    preds = model.predict(X)
    probs = model.predict_proba(X)

    # get class order from underlying estimator
    try:
        classes = model.classes_
    except AttributeError:
        classes = model.estimator.named_steps["clf"].classes_

    out = X.copy()
    out["Pred_NSP"] = preds
    for i, c in enumerate(classes):
        out[f"P(NSP={int(c)})"] = probs[:, i]

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote predictions → {args.out_csv}")

if __name__ == "__main__":
    main()
