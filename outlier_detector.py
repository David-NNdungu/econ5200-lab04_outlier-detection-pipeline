"""Outlier detection methods used in ECON 5200 Lab 4."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class OutlierDetector:
    """Run one of three outlier detection methods."""

    VALID_METHODS = (
        "tukey",
        "zscore",
        "isolation_forest"
    )

    def __init__(
        self,
        method="tukey",
        threshold=3.5,
        k=1.5,
        contamination=0.05,
        random_state=42
    ):
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"method must be one of {self.VALID_METHODS}, "
                f"got '{method}'"
            )

        if threshold <= 0:
            raise ValueError(
                f"threshold must be positive, got {threshold}"
            )

        if k <= 0:
            raise ValueError(
                f"k must be positive, got {k}"
            )

        if not 0 < contamination < 0.5:
            raise ValueError(
                "contamination must be between 0 and 0.5, "
                f"got {contamination}"
            )

        self.method = method
        self.threshold = threshold
        self.k = k
        self.contamination = contamination
        self.random_state = random_state

        self.mask_ = None
        self.n_outliers_ = None
        self.pct_outliers_ = None

    def fit_detect(self, data):
        """Return True for rows that are flagged as outliers."""

        if (
            self.method != "isolation_forest"
            and not isinstance(data, pd.Series)
        ):
            raise TypeError(
                f"'{self.method}' works on one column. "
                "Pass a pandas Series."
            )

        if self.method == "zscore":
            mask = self._zscore(data)

        elif self.method == "tukey":
            mask = self._tukey(data)

        else:
            mask = self._isolation_forest(data)

        self.mask_ = mask
        self.n_outliers_ = int(mask.sum())
        self.pct_outliers_ = float(mask.mean())

        return mask

    def summary(self):
        """Give a short summary of the results."""

        if self.mask_ is None:
            raise RuntimeError(
                "Call fit_detect() before asking for a summary."
            )

        if self.method == "zscore":
            params = {
                "threshold": self.threshold
            }

        elif self.method == "tukey":
            params = {
                "k": self.k
            }

        else:
            params = {
                "contamination": self.contamination
            }

        return {
            "method": self.method,
            "n_outliers": self.n_outliers_,
            "pct_outliers": round(
                self.pct_outliers_,
                4
            ),
            "params": params
        }

    def _zscore(self, series):
        """Calculate the modified Z-score using the median and MAD."""

        median = series.median()
        mad = np.median(
            np.abs(series - median)
        )

        if mad == 0:
            return pd.Series(
                False,
                index=series.index
            )

        modified_z = (
            0.6745
            * (series - median)
            / mad
        )

        return (
            np.abs(modified_z)
            > self.threshold
        )

    def _tukey(self, series):
        """Find values outside the Tukey fences."""

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_fence = q1 - self.k * iqr
        upper_fence = q3 + self.k * iqr

        return (
            (series < lower_fence)
            | (series > upper_fence)
        )

    def _isolation_forest(self, data):
        """Use Isolation Forest to find unusual rows."""

        if isinstance(data, pd.Series):
            data = data.to_frame()

        model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state
        )

        predictions = model.fit_predict(data)

        return pd.Series(
            predictions == -1,
            index=data.index
        )
