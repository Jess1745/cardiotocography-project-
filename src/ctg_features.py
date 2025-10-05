# ctg_features.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CTGEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.eng_cols_ = [
            "feat_tachycardia",
            "feat_bradycardia",
            "feat_variability_ratio",
            "feat_accel_decel_balance",
            "feat_decels_per_contraction",
            "feat_range",
            "feat_instability_proxy",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        def g(*cands):
            for c in cands:
                if c in X.columns:
                    return c
            return None

        baseline = g("lb","baseline","mean_fhr","fhr_baseline","b")
        ac = g("ac","accelerations")
        dl = g("dl","decelerations_late","late_decelerations")
        ds = g("ds","decelerations_short","short_decelerations")
        dp = g("dp","decelerations_prolonged","prolonged_decelerations")
        uc = g("uc","uterine_contractions")
        mstv = g("mstv")
        altv = g("altv")
        width = g("width")
        vmin = g("min")
        vmax = g("max")

        if baseline is not None:
            X["feat_tachycardia"] = (X[baseline] > 160).astype(int)
            X["feat_bradycardia"] = (X[baseline] < 110).astype(int)

        if altv is not None and mstv is not None:
            X["feat_variability_ratio"] = X[altv] / (X[mstv] + self.eps)

        decel_cols = [c for c in [dl, ds, dp] if c is not None]
        if ac is not None and decel_cols:
            X["feat_accel_decel_balance"] = (X[ac] + 1.0) / (X[decel_cols].sum(axis=1) + 1.0)

        if uc is not None and decel_cols:
            X["feat_decels_per_contraction"] = X[decel_cols].sum(axis=1) / (X[uc] + 1.0)

        if vmax is not None and vmin is not None:
            X["feat_range"] = X[vmax] - X[vmin]

        if width is not None and mstv is not None:
            X["feat_instability_proxy"] = X[width] / (X[mstv] + 1.0)

        # numeric cleanup
        num_cols = X.select_dtypes(include=[np.number]).columns
        X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)

        for c in self.eng_cols_:
            if c in X.columns and X[c].isna().any():
                X[c] = X[c].fillna(X[c].median())

        return X
