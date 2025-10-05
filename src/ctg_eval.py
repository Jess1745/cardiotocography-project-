import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import f1_score, balanced_accuracy_score, make_scorer

def make_cv(n_splits=5, seed=42):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

SCORING = {
    "macro_f1": make_scorer(f1_score, average="macro"),
    "bal_acc":  make_scorer(balanced_accuracy_score),
}

def evaluate(pipe, X, y, name, cv=None):
    if cv is None:
        cv = make_cv()
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1, return_train_score=False)
    print(f"{name:>14} | macro F1: {res['test_macro_f1'].mean():.3f} ± {res['test_macro_f1'].std():.3f}  | bal acc: {res['test_bal_acc'].mean():.3f}")
    return res