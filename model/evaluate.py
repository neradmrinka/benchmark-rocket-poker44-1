"""Honest cross-validation harness using the repo's EXACT reward() function.

Generalization is estimated with GroupKFold by source_date: a model is never
tested on a date it trained on (mirrors the docs' warning that live data differs
from any single benchmark date).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from reward_fn import reward


def remap_to_threshold(p: np.ndarray, t: float) -> np.ndarray:
    """Monotonic remap so decision threshold t maps to 0.5 (preserves AP/AUC).

    The validator computes preds = round(score), so the operating point must sit
    at 0.5. This piecewise-linear map keeps ranking identical (AP unchanged) while
    relocating the boundary."""
    t = float(min(max(t, 1e-6), 1 - 1e-6))
    out = np.where(p >= t,
                   0.5 + 0.5 * (p - t) / (1 - t),
                   0.5 * p / t)
    return np.clip(out, 0.0, 1.0)


def best_threshold(p: np.ndarray, y: np.ndarray):
    """Pick threshold maximizing reward() after remap, honoring the fpr<0.10 cliff."""
    best_t, best_r = 0.5, -1.0
    for t in np.linspace(0.05, 0.99, 95):
        r, _ = reward(remap_to_threshold(p, t), y)
        if r > best_r:
            best_r, best_t = r, t
    return best_t, best_r


def fpr_target_threshold(p_neg: np.ndarray, target_fpr: float) -> float:
    """Threshold = quantile of NEGATIVE (human) scores giving ~target_fpr.

    Choosing the boundary relative to the human score distribution is far more
    robust to cross-date shift than an absolute threshold, and directly controls
    the fpr that the human-safety cliff punishes."""
    if len(p_neg) == 0:
        return 0.5
    q = 1.0 - float(target_fpr)
    return float(np.quantile(p_neg, min(max(q, 0.0), 1.0)))


def evaluate(make_model, X, y, dates, n_splits=6, label="", target_fpr=0.05):
    """Per-fold evaluation (faithful to how the validator scores one batch).

    For each held-out fold: fit on the other dates, pick a safe threshold on the
    TRAINING humans, remap so it sits at 0.5, then score that fold's test batch
    with the repo reward(). Aggregate across folds."""
    Xv = X.values.astype(float)
    gkf = GroupKFold(n_splits=n_splits)
    fold_ap, fold_reward, fold_fpr, fold_recall, fold_t = [], [], [], [], []
    oof = np.zeros(len(y))
    for tr, te in gkf.split(Xv, y, groups=dates):
        model = make_model()
        model.fit(Xv[tr], y[tr])
        p_te = model.predict_proba(Xv[te])[:, 1]
        oof[te] = p_te
        # NESTED calibration: out-of-fold scores within the TRAINING dates give an
        # honest (non-overconfident) picture of where humans land on unseen dates.
        inner = GroupKFold(n_splits=min(5, len(set(dates[tr]))))
        p_tr_oof = cross_val_predict(
            make_model(), Xv[tr], y[tr], groups=dates[tr],
            cv=inner, method="predict_proba")[:, 1]
        t = fpr_target_threshold(p_tr_oof[y[tr] == 0], target_fpr)
        scores = remap_to_threshold(p_te, t)
        r, res = reward(scores, y[te])
        if np.any(y[te] == 1) and np.any(y[te] == 0):
            fold_ap.append(average_precision_score(y[te], p_te))
        fold_reward.append(r)
        fold_fpr.append(res["fpr"])
        fold_recall.append(res["bot_recall"])
        fold_t.append(t)
    out = {
        "label": label,
        "ap": float(np.mean(fold_ap)),          # mean per-fold AP
        "ap_pooled": average_precision_score(y, oof),
        "auc": roc_auc_score(y, oof),
        "reward": float(np.mean(fold_reward)),   # mean per-fold reward (the headline)
        "reward_std": float(np.std(fold_reward)),
        "fpr": float(np.mean(fold_fpr)),
        "recall": float(np.mean(fold_recall)),
        "n_zero_folds": int(sum(1 for r in fold_reward if r == 0.0)),
        "target_fpr": target_fpr,
    }
    return out, oof


def fmt(out):
    return (f"{out['label']:28s} | AP={out['ap']:.4f} AUCp={out['auc']:.4f} "
            f"| reward={out['reward']:.4f}±{out['reward_std']:.3f} "
            f"(fpr={out['fpr']:.3f} recall={out['recall']:.3f}) "
            f"| zero_folds={out['n_zero_folds']}/6")


if __name__ == "__main__":
    from sklearn.ensemble import HistGradientBoostingClassifier
    from featurize import build_matrix

    X, y, dates, splits = build_matrix()
    print(f"X={X.shape}  dates={len(set(dates))}\n")

    def hgb():
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300,
            l2_regularization=1.0, random_state=0)

    print("Scan deploy target_fpr (threshold robustness vs the 0.10 cliff):")
    for tf in (0.02, 0.03, 0.05, 0.08):
        out, _ = evaluate(hgb, X, y, dates, label=f"HGB target_fpr={tf}", target_fpr=tf)
        print(fmt(out))
