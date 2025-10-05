import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (f1_score, balanced_accuracy_score, make_scorer,
                             classification_report, confusion_matrix, roc_curve,
                             auc, precision_recall_curve, average_precision_score)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.base import clone

# your module with preprocessing + CTGEngineer
from ctg_prep import make_preprocess_trees, make_preprocess_linear

# --- utils ---
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # make keys tolerant to casing / whitespace - keeps your original names lowercase internally
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    df = df.rename(columns={c: c.lower() for c in df.columns})
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="CTG_clean.csv", help="Training CSV (must contain NSP)")
    ap.add_argument("--out_models", default="models", help="Where to save weights")
    ap.add_argument("--out_reports", default="reports", help="Where to save reports & figures")
    args = ap.parse_args()

    OUT_MODELS = Path(args.out_models); OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_REP = Path(args.out_reports); (OUT_REP / "figures").mkdir(parents=True, exist_ok=True)
    FIG_DIR = OUT_REP / "figures"

    # --- DATA ---
    df = pd.read_csv(args.train_csv)
    df = normalize_cols(df)
    if "nsp" not in df.columns:
        raise SystemExit("Training CSV must include 'NSP' (any casing).")
    df = df[df["nsp"].notna()].copy()
    df["nsp"] = df["nsp"].astype(int)
    y = df["nsp"].values
    X = df.drop(columns=["nsp", "filename", "segfile"], errors="ignore")

    # --- CV + scoring ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "macro_f1": make_scorer(f1_score, average="macro"),
        "bal_acc": make_scorer(balanced_accuracy_score)
    }

    def evaluate(pipe, X, y, name):
        res = cross_validate(pipe, X, y, cv=skf, scoring=scoring, n_jobs=-1, return_train_score=False)
        print(f"{name:>14} | macro F1: {res['test_macro_f1'].mean():.3f} ± {res['test_macro_f1'].std():.3f} | "
              f"bal acc: {res['test_bal_acc'].mean():.3f}")
        return res

    # --- MODELS ---
    pipe_dummy = Pipeline([("prep", make_preprocess_linear()), ("clf", DummyClassifier(strategy="stratified", random_state=42))])
    pipe_lr    = Pipeline([("prep", make_preprocess_linear()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", multi_class="multinomial", solver="lbfgs"))])
    pipe_svm   = Pipeline([("prep", make_preprocess_linear()), ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced"))])
    pipe_rf    = Pipeline([("prep", make_preprocess_trees()),  ("clf", RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", random_state=42))])

    # Evaluate ladder
    res_dummy = evaluate(pipe_dummy, X, y, "Dummy")
    res_lr    = evaluate(pipe_lr,    X, y, "LogReg")
    res_svm   = evaluate(pipe_svm,   X, y, "SVM RBF")
    res_rf    = evaluate(pipe_rf,    X, y, "RandForest")

    # Pick best by CV macro-F1
    candidates = [
        ("Dummy", pipe_dummy, res_dummy),
        ("LogReg (base)", pipe_lr, res_lr),
        ("SVM RBF (base)", pipe_svm, res_svm),
        ("RF (base)", pipe_rf, res_rf),
    ]
    best_name, best_pipe, _ = max(candidates, key=lambda t: t[2]["test_macro_f1"].mean())
    print("Selected best:", best_name)

    # OOF predictions for figures
    classes = np.sort(np.unique(y))
    label_names = ["Normal","Suspect","Pathologic"]  # assumes 1,2,3
    oof_prob = cross_val_predict(best_pipe, X, y, cv=skf, method="predict_proba", n_jobs=-1)
    oof_pred = classes[oof_prob.argmax(axis=1)]

    # Confusion matrix
    cm = confusion_matrix(y, oof_pred, labels=classes)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation="nearest", cmap="viridis")
    plt.title("Confusion Matrix (OOF)")
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, label_names); plt.yticks(ticks, label_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color=("white" if cm[i,j] > cm.max()/2 else "black"))
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(FIG_DIR/"confusion_matrix_oof.png", dpi=160); plt.close()

    # Per-class report CSV
    report = classification_report(y, oof_pred, labels=classes, target_names=label_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(OUT_REP/"per_class_f1_oof.csv")

    # ROC / PR (OvR)
    plt.figure(figsize=(8,7))
    for i, cls in enumerate(classes):
        y_bin = (y == cls).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, oof_prob[:, i])
        plt.plot(fpr, tpr, label=f"{label_names[i]} (AUC={auc(fpr,tpr):.3f})")
    plt.plot([0,1],[0,1],"--", color="r"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (OvR)"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR/"roc_ovr.png", dpi=160); plt.close()

    plt.figure(figsize=(8,7))
    aucs, aps = [], []
    for i, cls in enumerate(classes):
        y_bin = (y == cls).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, oof_prob[:, i])
        ap = average_precision_score(y_bin, oof_prob[:, i]); aps.append(ap)
        plt.plot(rec, prec, label=f"{label_names[i]} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (OvR)"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR/"pr_ovr.png", dpi=160); plt.close()

    pd.DataFrame({
        "class_id": classes.astype(int),
        "class_name": label_names,
        "AUC": [np.nan]*len(classes),
        "AP": np.round(aps, 4),
    }).to_csv(OUT_REP/"ovr_auc_ap.csv", index=False)

    # Calibration + ECE
    from sklearn.calibration import calibration_curve
    def expected_calibration_error(y_true_bin, y_prob, n_bins=15):
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(y_prob, bins) - 1
        ece = 0.0
        for b in range(n_bins):
            mask = (bin_ids == b)
            if not np.any(mask): 
                continue
            ece += abs((y_true_bin[mask]==1).mean() - y_prob[mask].mean()) * mask.mean()
        return ece

    rows = []
    for i, cls in enumerate(classes):
        y_bin = (y == cls).astype(int)
        frac_pos, mean_pred = calibration_curve(y_bin, oof_prob[:, i], n_bins=15, strategy="uniform")
        ece_k = expected_calibration_error(y_bin, oof_prob[:, i], n_bins=15)
        rows.append({"class_id": int(cls), "class_name": label_names[i], "ECE": round(ece_k, 4)})
        # plot per-class
        plt.figure(figsize=(6,6))
        plt.plot([0, 1], [0, 1], "--", label="Perfect")
        plt.plot(mean_pred, frac_pos, marker="o", linewidth=1, label=f"{label_names[i]}  (ECE={ece_k:.03f})")
        plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
        plt.title(f"Calibration — {label_names[i]}"); plt.legend(); plt.tight_layout()
        plt.savefig(FIG_DIR / f"calibration_class{cls}.png", dpi=160); plt.close()
    calib_df = pd.DataFrame(rows); calib_df["ECE_macro"] = calib_df["ECE"].mean()
    calib_df.to_csv(OUT_REP/"calibration_ece_oof.csv", index=False)

    # Save comparison table
    rows_cmp = []
    def addrow(name, res):
        rows_cmp.append({
            "model": name,
            "cv_macro_f1_mean": float(res["test_macro_f1"].mean()),
            "cv_macro_f1_std":  float(res["test_macro_f1"].std()),
            "cv_balacc_mean":   float(res["test_bal_acc"].mean()),
            "cv_balacc_std":    float(res["test_bal_acc"].std()),
        })
    for name, pipe, res in candidates:
        addrow(name, res)
    pd.DataFrame(rows_cmp).sort_values("cv_macro_f1_mean", ascending=False)\
        .to_csv(OUT_REP/"model_comparison.csv", index=False)

    # Calibrate & save final model
    calibrated = CalibratedClassifierCV(estimator=best_pipe, method="sigmoid", cv=5)
    calibrated.fit(X, y)
    joblib.dump(calibrated, OUT_MODELS/"ctg_best_calibrated.joblib")
    print("[OK] Saved ->", OUT_MODELS/"ctg_best_calibrated.joblib")

    # Permutation importance (global)
    best_pipe_fit = clone(best_pipe).fit(X, y)
    r = permutation_importance(best_pipe_fit, X, y, scoring="f1_macro", n_repeats=10, random_state=42, n_jobs=-1)
    pi = pd.DataFrame({"feature": X.columns, "importance": r.importances_mean, "std": r.importances_std})\
         .sort_values("importance", ascending=False)
    pi.to_csv(OUT_REP/"permutation_importance_original.csv", index=False)
    plt.figure(figsize=(8,8))
    plt.barh(pi.head(20)["feature"][::-1], pi.head(20)["importance"][::-1])
    plt.xlabel("Permutation importance (macro-F1 drop)")
    plt.title("Top features by permutation importance"); plt.tight_layout()
    plt.savefig(FIG_DIR/"permutation_importance_top20_original.png", dpi=160); plt.close()

    # Robustness (1% noise)
    rng = np.random.default_rng(42)
    X_num = X.select_dtypes(include=[np.number])
    X_noise = X.copy()
    X_noise[X_num.columns] = X_num + rng.normal(0, X_num.std(ddof=0)*0.01, size=X_num.shape)
    y_pred0 = best_pipe_fit.predict(X)
    y_pred1 = best_pipe_fit.predict(X_noise)
    delta = f1_score(y, y_pred1, average="macro") - f1_score(y, y_pred0, average="macro")
    (OUT_REP/"robustness.txt").write_text(f"Delta macro-F1 with 1% numeric noise: {delta:.4f}\n")
    print(f"[OK] Robustness Δ macro-F1 = {delta:.4f}")

    try:
        import torch
        clf = best_pipe_fit.named_steps.get("clf", None)

        if isinstance(clf, LogisticRegression):
            state = {
                "type": "logistic_regression",
                "coef_": torch.tensor(clf.coef_, dtype=torch.float32),
                "intercept_": torch.tensor(clf.intercept_, dtype=torch.float32),
                "classes_": torch.tensor(clf.classes_, dtype=torch.int64),
                "note": "Pure LR weights; preprocessing (scaling, imputation) handled separately in sklearn pipeline."
            }

        elif isinstance(clf, SVC):
            # SVM doesn’t have traditional linear weights, so we store support vectors instead
            state = {
                "type": "svm_rbf",
                "support_vectors_": torch.tensor(clf.support_vectors_, dtype=torch.float32),
                "dual_coef_": torch.tensor(clf.dual_coef_, dtype=torch.float32),
                "intercept_": torch.tensor(clf.intercept_, dtype=torch.float32),
                "classes_": torch.tensor(clf.classes_, dtype=torch.int64),
                "note": "SVM (RBF) model — no explicit feature weights, stores kernel support vectors."
            }

        elif isinstance(clf, RandomForestClassifier):
            state = {
                "type": "random_forest",
                "n_estimators": clf.n_estimators,
                "feature_importances_": torch.tensor(clf.feature_importances_, dtype=torch.float32),
                "note": "RandomForest feature importances only — full trees stored in sklearn pipeline."
            }

        else:
            state = {
                "type": str(type(clf)),
                "note": "Unsupported model type for direct tensor export — saved metadata only."
            }

        torch.save(state, OUT_MODELS / "ctg_best_linear.pt")
        print("[OK] Saved ->", OUT_MODELS / "ctg_best_linear.pt")

    except Exception as e:
        print("[WARN] .pt export skipped (install torch or check model type). Err:", e)


if __name__ == "__main__":
    main()