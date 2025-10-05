# ctg_prep.py
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# If ctg_features.py is in the same folder, use this import:
from ctg_features import CTGEngineer

numeric_selector = make_column_selector(dtype_include=np.number)

def make_preprocess_trees():
    return Pipeline([
        ("eng", CTGEngineer()),
        ("num", ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                ]), numeric_selector),
            ],
            remainder="drop"
        )),
    ])

def make_preprocess_linear():
    return Pipeline([
        ("eng", CTGEngineer()),
        ("num", ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]), numeric_selector),
            ],
            remainder="drop"
        )),
    ])
