"""The composer: TabPFN-3 in context over previous patients' labelled outputs (no training), 5 seeds averaged, missing inputs
imputed with the context's median. The patient and core thresholds are fixed on the context by inner cross-validation; the
targeting threshold is given at inference (run.py --threshold, default TARGETING_THRESHOLD)."""

from __future__ import annotations

import os

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

SEEDS = (0, 1, 2, 3, 4)
FLAGGED_SHARE = 0.30  # the core composer flags the top 30% of cores (the operators' biopsy rate in the paper's cohort)
TARGETING_THRESHOLD = 0.14  # the paper's targeting threshold: P(csPCa) above which its composer flags the top ~30% of cores


def fit_tabpfn(X, y, seed: int = 0):
    import torch
    from tabpfn import TabPFNClassifier

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return TabPFNClassifier(
        device="cuda" if torch.cuda.is_available() else "cpu", random_state=seed, ignore_pretraining_limits=True
    ).fit(X, y)


def positive_proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def impute(X: np.ndarray, median: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(X), median, X)


def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(np.clip(thr[(tpr - fpr).argmax()], 0.02, 0.98))


def rate_threshold(p, positive_rate: float) -> float:
    """The threshold that flags the top `positive_rate` of the scores."""
    return float(np.quantile(p, 1 - positive_rate))


class TabPFNComposer:
    def __init__(self, keys: list[str], X: np.ndarray, y: np.ndarray, seeds=SEEDS, threshold: float | None = None):
        """X: the context (rows = previous patients or cores, columns = `keys`, NaN = missing); y: their labels."""
        self.keys, self.seeds = list(keys), tuple(seeds)
        self.median = np.nan_to_num(np.nanmedian(X, axis=0), nan=0.0)
        self.X, self.y = impute(X, self.median), np.asarray(y, int)
        self.threshold = threshold
        self._models = None

    @property
    def models(self):
        if self._models is None:
            self._models = [fit_tabpfn(self.X, self.y, s) for s in self.seeds]
        return self._models

    def proba(self, X: np.ndarray) -> np.ndarray:
        p = np.mean([positive_proba(m, impute(X, self.median)) for m in self.models], 0)
        return np.clip(p, 1e-4, 1 - 1e-4)

    def inner_oof(self, n_splits: int = 5) -> np.ndarray:
        """Out-of-fold probabilities on the context (to fix the threshold without the held-out patients)."""
        oof = np.zeros(len(self.y))
        for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=0).split(self.X, self.y):
            oof[te] = np.mean([positive_proba(fit_tabpfn(self.X[tr], self.y[tr], s), self.X[te]) for s in self.seeds], 0)
        return oof

    def predict(self, features: dict, positive: str, negative: str) -> dict:
        """Risk, decision at the threshold, which inputs were imputed, and each input's contribution (change in log-odds when
        it is replaced by the context median)."""
        x = np.array([features.get(k, np.nan) for k in self.keys], float)
        imputed = [k for k, v in zip(self.keys, x) if np.isnan(v)]
        x = impute(x, self.median)
        X = np.vstack([x] + [np.where(np.arange(len(x)) == j, self.median, x) for j in range(len(x))])
        p = self.proba(X)
        lo = np.log(p / (1 - p))
        r, t = float(p[0]), float(self.threshold)
        contrib = sorted(zip(self.keys, (lo[0] - lo[1:]).tolist()), key=lambda kv: -abs(kv[1]))
        return {
            "risk": round(r, 4),
            "threshold": round(t, 3),
            "decision": positive if r >= t else negative,
            "confidence": "high" if abs(r - t) >= 0.3 else "moderate" if abs(r - t) >= 0.15 else "borderline",
            "inputs_imputed": imputed,
            "top_contributions": [(k, round(v, 2)) for k, v in contrib[:4]],
        }
