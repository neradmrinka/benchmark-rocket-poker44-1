"""Serving path for the rocket miner: chunks -> one bot-risk score per chunk.

Loads the trained RocketEnsemble, extracts the phasberg + v2 feature views (the SAME
extractors the trainer used — d0_features is imported by both, so train == serve),
fuses the four components with this variant's rule, and returns one score per chunk.

On the two post-processing steps below: the reward is 0.75*AP +
0.25*recall@fpr<=0.05, and BOTH terms read only the ORDER of the scores. So neither
step can move the reward — they exist so that `predictions` (score >= 0.5) means
something sane. Both are strictly rank-preserving; keeping them that way is the
invariant to protect if you ever touch this file.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import live_probe
from d0_drse import DRSE  # noqa: F401  (DRSE instances live inside the pickle)
from d0_features import phasberg_dict, v2_dict
from rocket_ensemble import RocketEnsemble  # noqa: F401  (needed to unpickle model.pkl)

_ART = Path(__file__).resolve().parent / "artifacts"

# Cap how many chunks may be called bot in one validator batch. Ranking (and so the
# reward) is untouched; this only keeps `predictions` from claiming a runaway
# positive rate.
_MAX_POS_FRAC = float(os.environ.get("POKER44_MAX_POS_FRAC", "0.15"))


def _apply_batch_safety_budget(scores: np.ndarray, max_frac: float) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    n = s.size
    if n == 0 or max_frac >= 1.0:
        return s
    k = int(np.floor(max_frac * n))
    order = np.argsort(-s, kind="stable")
    allowed = {int(i) for i in order[:k] if s[i] >= 0.5}
    out = s.copy()
    squeeze = [int(i) for i in order if int(i) not in allowed]
    m = len(squeeze)
    for rank, idx in enumerate(squeeze):
        out[idx] = 0.499 * (1.0 - rank / max(m - 1, 1))
    return np.clip(out, 0.0, 1.0)


def _remap_to_threshold(p: np.ndarray, t: float) -> np.ndarray:
    """Monotonic map putting the deploy threshold t at 0.5 (order preserved)."""
    t = float(min(max(t, 1e-6), 1 - 1e-6))
    out = np.where(p >= t, 0.5 + 0.5 * (p - t) / (1 - t), 0.5 * p / t)
    return np.clip(out, 0.0, 1.0)


class Poker44Model:
    def __init__(self, art_dir: Path = _ART):
        art_dir = Path(art_dir)
        model_path = art_dir / "model.pkl"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"no trained model at {model_path} — run `python3 train_rocket.py` "
                f"(or `python3 autopilot.py`) before serving"
            )
        with model_path.open("rb") as fh:
            self.ens: RocketEnsemble = pickle.load(fh)
        with (art_dir / "meta.json").open() as fh:
            self.meta: Dict[str, Any] = json.load(fh)
        self.threshold: float = float(self.meta["deploy_threshold"])
        self.cph = self.ens.cols_ph
        self.cv2 = self.ens.cols_v2

    def _matrices(self, chunks):
        ph = np.array([[float(d.get(c, 0.0)) for c in self.cph]
                       for d in (phasberg_dict(c) for c in chunks)], dtype=float)
        v2 = np.array([[float(d.get(c, 0.0)) for c in self.cv2]
                       for d in (v2_dict(c) for c in chunks)], dtype=float)
        return ph, v2

    def score_chunks(self, chunks: List[List[Dict[str, Any]]]) -> List[float]:
        if not chunks:
            return []
        ph, v2 = self._matrices(chunks)
        live_probe.record_batch(v2, n_chunks=len(chunks))

        p = self.ens.score(ph, v2)
        scores = _remap_to_threshold(p, self.threshold)
        scores = _apply_batch_safety_budget(scores, _MAX_POS_FRAC)
        return [0.1 if not chunk else round(float(s), 6)
                for chunk, s in zip(chunks, scores)]


_SINGLETON: Poker44Model | None = None


def get_model() -> Poker44Model:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = Poker44Model()
    return _SINGLETON


if __name__ == "__main__":
    from dataset import load_examples

    examples = load_examples()
    model = get_model()
    scores = model.score_chunks([e.hands for e in examples[:12]])
    print(f"variant={model.meta.get('variant')} weights={model.meta.get('weights')}")
    print(f"threshold={model.threshold:.4f} cv_ap={model.meta['cv_ap']:.4f} "
          f"cv_reward={model.meta['cv_reward']:.4f}")
    print("scores:", [round(s, 3) for s in scores])
