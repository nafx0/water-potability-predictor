from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names: Optional[Sequence[str]] = None):
        self.feature_names = list(feature_names) if feature_names is not None else None

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        elif self.feature_names is not None:
            self.feature_names_ = list(self.feature_names)
        else:
            n_features = X.shape[1]
            self.feature_names_ = [f"x{i}" for i in range(n_features)]
        return self

    def transform(self, X):
        import pandas as pd

        X_df = pd.DataFrame(X, columns=self.feature_names_)

        if {"Solids", "Hardness"}.issubset(X_df.columns):
            X_df["solids_to_hardness"] = X_df["Solids"] / (X_df["Hardness"] + 1)
        if {"Chloramines", "ph"}.issubset(X_df.columns):
            X_df["chloramines_to_ph"] = X_df["Chloramines"] / (X_df["ph"] + 1)

        return X_df.values

class IQRClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X)
        self.bounds_ = []
        for i in range(X.shape[1]):
            q1 = np.percentile(X[:, i], 25)
            q3 = np.percentile(X[:, i], 75)
            iqr = q3 - q1
            self.bounds_.append((q1 - 1.5*iqr, q3 + 1.5*iqr))
        return self

    def transform(self, X):
        X = np.asarray(X).copy()
        for i, (low, high) in enumerate(self.bounds_):
            X[:, i] = np.clip(X[:, i], low, high)
        return X