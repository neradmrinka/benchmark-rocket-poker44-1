"""Stack-calibration helpers.

Active:
  * :class:`BlendedIsotonicCalibrator` — the isotonic stack calibrator, dulled by
    blending its (step-shaped) output back toward the raw score so plateaus
    smooth out and ranking resolution is preserved.

Retained for backward-compatibility only (do NOT wire into new training code):
  * :class:`BlendedQuantileCalibrator` — the removed quantile calibrator, kept so
    previously-saved ``.joblib`` artifacts that pickled it inside their
    ``StackedEnsemble`` can still be unpickled and scored.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import QuantileTransformer


class BlendedIsotonicCalibrator:
    """Isotonic stack calibration, dulled by blending toward identity.

    Pure isotonic regression is piecewise-constant: many distinct raw scores
    collapse onto the same plateau (sharp steps + abrupt jumps), erasing
    within-plateau ranking resolution. Blending the isotonic output with the raw
    score pulls it back toward identity, smoothing the steps while staying
    monotone (a convex combination of two monotone functions is monotone, and the
    identity term makes it strictly increasing so ties are broken)::

        out = blend * isotonic(raw) + (1 - blend) * raw

    ``blend=1.0`` -> pure isotonic (sharpest); ``blend=0.0`` -> passthrough
    (no calibration). Lower ``blend`` = duller / smoother.
    """

    def __init__(self, blend: float = 0.5) -> None:
        self.blend = float(max(0.0, min(1.0, blend)))
        self._iso: Optional[IsotonicRegression] = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "BlendedIsotonicCalibrator":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(
            np.asarray(scores, dtype=float).ravel(),
            np.asarray(labels, dtype=float).ravel(),
        )
        self._iso = iso
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(scores, dtype=float).ravel(), 0.0, 1.0)
        if self._iso is None:
            return values
        isotonic = np.asarray(self._iso.transform(values), dtype=float)
        mixed = self.blend * isotonic + (1.0 - self.blend) * values
        return np.clip(mixed, 0.0, 1.0)


class BlendedQuantileCalibrator:
    """Monotone score spreader for collapsed stacked probabilities.

    Deprecated/unused: kept only for unpickling legacy saved models. See the
    module docstring.
    """

    def __init__(self, blend: float = 0.9, max_quantiles: int = 256) -> None:
        self.blend = float(max(0.0, min(1.0, blend)))
        self.max_quantiles = int(max(8, max_quantiles))
        self._qt: Optional[QuantileTransformer] = None

    def fit(self, scores: np.ndarray) -> "BlendedQuantileCalibrator":
        values = np.asarray(scores, dtype=float).reshape(-1, 1)
        n_quantiles = int(max(8, min(self.max_quantiles, len(values))))
        qt = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="uniform",
            subsample=max(len(values), 1000),
            random_state=42,
        )
        qt.fit(values)
        self._qt = qt
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=float).reshape(-1, 1)
        if self._qt is None:
            return np.clip(values.ravel(), 0.0, 1.0)
        uniformized = self._qt.transform(values).ravel()
        base = np.clip(values.ravel(), 0.0, 1.0)
        mixed = self.blend * uniformized + (1.0 - self.blend) * base
        return np.clip(mixed, 0.0, 1.0)
