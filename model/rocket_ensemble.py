"""rocket-r1 — weighted RANK fusion of four decorrelated components.

    r1 = w_stack*rank(stack) + w_mono*rank(mono) + w_mlp*rank(mlp) + w_drse*rank(drse)

Components (unchanged from the dragon-0 recipe — this is a fine-tune, not a rewrite):
  stack  StackingClassifier(LGBM+XGB+CatBoost+ET+RF -> LR meta) on the phasberg view
  mono   sign-constrained XGBoost vote on the phasberg view (drift-stable)
  mlp    PCA -> MLP vote on the v2+phasberg UNION (a non-tree family; decorrelated)
  drse   drift-robust subspace ensemble on the v2 view (a feature space no other
         component sees)

Why rank fusion: the reward is 0.75*AP + 0.25*recall@fpr<=0.05, and both terms read
only the ORDER of the scores. Rank averaging is therefore calibration-free — a
component that is badly scaled but correctly ordered still contributes fully — and it
cannot be dragged around by one over-confident model the way logit averaging can.
(r2 deliberately takes the opposite bet; the two are meant to differ.)

blend() is the ONE definition of the aggregation. train_rocket.py's walk-forward calls
this exact function to score candidate weights, and RocketEnsemble.score() calls it to
serve. dragon-0 kept two hand-synced copies (D0Ensemble.score and train's blend()); a
drift between them silently invalidates the whole walk-forward, so there is now one.

Picklable. stack/mono take phasberg; mlp takes hstack([v2, phasberg]); drse takes v2.
"""

import numpy as np

from variant import W_PRIOR

PARTS = ("stack", "mono", "mlp", "drse")


def _rank(scores):
    """Map scores to evenly spaced ranks in [0, 1] (ties broken stably by index)."""
    s = np.asarray(scores, dtype=float)
    if s.size <= 1:
        return s.astype(float)
    return np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float) / (s.size - 1)


def blend(parts, weights=None):
    """Weighted mean of per-component ranks.

    parts: {component name -> P(bot) array}, all the same length.

    A single-chunk batch has no ranking to speak of (every rank would collapse to
    the same value), so we fall back to the weighted mean of the raw probabilities
    there — a validator batch is ~80 chunks, this only guards hand-probes.
    """
    w = dict(weights or W_PRIOR)
    total = float(sum(w[k] for k in PARTS))
    if total <= 0:
        raise ValueError(f"blend weights must sum > 0, got {w}")

    n = np.asarray(parts[PARTS[0]], dtype=float).size
    transform = (lambda v: np.asarray(v, dtype=float)) if n <= 1 else _rank

    out = np.zeros(n, dtype=float)
    for name in PARTS:
        out += w[name] * transform(parts[name])
    return out / total


class RocketEnsemble:
    """The served rocket-r1 model. Holds its four fitted components + column order."""

    variant = "r1"
    fusion = "weighted-rank"

    def __init__(self, stack, mono, mlp, drse, cols_ph, cols_v2, weights=None):
        self.stack = stack
        self.mono = mono
        self.mlp = mlp
        self.drse = drse
        self.cols_ph = cols_ph
        self.cols_v2 = cols_v2
        self.weights = dict(weights) if weights else dict(W_PRIOR)

    def components(self, Xph, Xv2):
        """Per-component P(bot). Kept separate from blend() so the trainer can
        fit once per walk-forward date and then score every candidate weighting
        off the same predictions."""
        Xph = np.asarray(Xph, dtype=float)
        Xv2 = np.asarray(Xv2, dtype=float)
        Xun = np.hstack([Xv2, Xph])
        return {
            "stack": self.stack.predict_proba(Xph)[:, 1],
            "mono": self.mono.predict_proba(Xph)[:, 1],
            "mlp": self.mlp.predict_proba(Xun)[:, 1],
            "drse": self.drse.predict_proba(Xv2)[:, 1],
        }

    def score(self, Xph, Xv2):
        """Xph: phasberg matrix in cols_ph order; Xv2: v2 matrix in cols_v2 order."""
        return blend(self.components(Xph, Xv2), self.weights)
