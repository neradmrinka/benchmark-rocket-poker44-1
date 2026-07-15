"""Drift-robust subspace ensemble (DRSE) on the v2 feature space — pocker_d0's 4th component.

Bagged ExtraTrees / HistGBM, each fit on a random FEATURE subspace + bootstrap rows. Averaging
over subspaces makes the model robust to per-feature distribution drift between the benchmark
and the live feed (no single feature can dominate). Runs on v2 (sanitization-invariant) features,
which the phasberg-based stack/mono do NOT use — so it adds a genuinely decorrelated, different
feature-space vote to d0's blend. Picklable.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier


class DRSE:
    def __init__(self, n=10, ff=0.7, seed=0):
        self.n, self.ff, self.seed = n, ff, seed

    def fit(self, X, y):
        X = np.asarray(X, float)
        rng = np.random.RandomState(self.seed)
        nf = X.shape[1]
        k = max(1, int(self.ff * nf))
        self.mem = []
        for b in range(self.n):
            fi = np.sort(rng.choice(nf, k, replace=False))
            rows = rng.choice(len(X), len(X), replace=True)
            m = (ExtraTreesClassifier(300, max_depth=14, min_samples_leaf=2, n_jobs=4,
                                      random_state=b, class_weight="balanced_subsample")
                 if b % 2 == 0 else
                 HistGradientBoostingClassifier(max_depth=4, learning_rate=0.04, max_iter=350,
                                                l2_regularization=2.0, random_state=b))
            m.fit(X[np.ix_(rows, fi)], y[rows])
            self.mem.append((fi, m))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        P = np.column_stack([m.predict_proba(X[:, fi])[:, 1] for fi, m in self.mem])
        a = P.mean(1)
        return np.column_stack([1.0 - a, a])
