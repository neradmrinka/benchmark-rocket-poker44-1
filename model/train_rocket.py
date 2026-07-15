"""Train rocket-r1 — weighted rank fusion, blend weights fitted by walk-forward.

    python3 train_rocket.py          # writes artifacts/model.pkl + meta.json

What this does that dragon-0's train_d0_ensemble.py did not:

1. BLEND WEIGHTS ARE FITTED, NOT GUESSED. d0 hard-coded 0.30/0.20/0.30/0.20. Here
   each walk-forward date fits the four components once, keeps their raw
   out-of-fold probabilities, and then every candidate in variant.W_GRID is scored
   against that same pool — so the search costs one blend evaluation per candidate,
   not one refit. The prior only loses its seat if a rival beats it by more than
   W_SELECT_MARGIN, because a few dates of held-out data cannot resolve a 0.001
   difference. d0's own point is in the grid, so the worst case is that we
   reproduce d0.

2. ONE DEFINITION OF THE BLEND. The walk-forward and the served model both call
   rocket_ensemble.blend(). d0 kept a second copy inside the trainer; if the two
   ever drifted apart, the walk-forward would have been measuring a model that is
   not the one being shipped, and nothing would have flagged it.

3. NO HARD-CODED PATHS. d0's trainer carried sys.path.insert("/root/my_pocker/...")
   from the machine it was written on, which simply fails anywhere else.

4. A DRIFT BASELINE. Training column statistics are written to
   artifacts/drift_baseline.json so autopilot.py can compare them against what the
   validator actually sends (see live_probe.py) and say whether the model is still
   being asked the question it was trained on.

Train == serve: every hand is pushed through the validator's own
prepare_hand_for_miner() before featurisation, so the model learns the sanitized
distribution it will be scored on — not the raw benchmark.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
REPO_DIR = MODEL_DIR.parent
for _p in (str(MODEL_DIR), str(REPO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import variant
from d0_drse import DRSE
from d0_features import phasberg_dict, v2_dict
from dataset import load_examples
from evaluate import fpr_target_threshold
from poker44.validator.payload_view import prepare_hand_for_miner
from reward_fn import reward
from rocket_ensemble import PARTS, RocketEnsemble, blend

ART = MODEL_DIR / "artifacts"
TARGET_FPR = 0.04
NJ = int(os.environ.get("POKER44_TRAIN_JOBS", "4"))
WF = int(os.environ.get("POKER44_WF_POINTS", "4"))


def sanitize(hands):
    """Project raw benchmark hands through the validator's miner-visible view."""
    out = []
    for hand in hands:
        try:
            out.append(prepare_hand_for_miner(hand))
        except Exception:
            out.append(hand)
    return out


def mat(chunks, fn, cols=None):
    frame = pd.DataFrame([fn(c) for c in chunks]).fillna(0.0)
    if cols is None:
        cols = sorted(frame.columns)
    return frame.reindex(columns=cols, fill_value=0.0).values, cols


# --------------------------------------------------------------------------- #
# components (hyper-parameters come from variant.py — that is what makes this
# miner's model different from its siblings')
# --------------------------------------------------------------------------- #
def make_stack():
    cfg = variant.STACK
    base = [
        ("lgb", lgb.LGBMClassifier(n_estimators=cfg["lgb_n"], learning_rate=cfg["lgb_lr"],
                                   num_leaves=cfg["lgb_leaves"], n_jobs=NJ,
                                   random_state=variant.SEED, verbose=-1)),
        ("xgb", xgb.XGBClassifier(n_estimators=cfg["xgb_n"], learning_rate=cfg["xgb_lr"],
                                  max_depth=cfg["xgb_depth"], tree_method="hist", n_jobs=NJ,
                                  random_state=variant.SEED, eval_metric="logloss")),
        ("cat", cb.CatBoostClassifier(iterations=cfg["cat_n"], learning_rate=cfg["cat_lr"],
                                      depth=cfg["cat_depth"], verbose=0, thread_count=NJ,
                                      random_seed=variant.SEED)),
        ("et", ExtraTreesClassifier(n_estimators=cfg["et_n"], max_depth=cfg["et_depth"],
                                    n_jobs=NJ, random_state=variant.SEED,
                                    class_weight="balanced_subsample")),
        ("rf", RandomForestClassifier(n_estimators=cfg["rf_n"], max_depth=cfg["rf_depth"],
                                      n_jobs=NJ, random_state=variant.SEED,
                                      class_weight="balanced_subsample")),
    ]
    return StackingClassifier(
        base, final_estimator=LogisticRegression(C=variant.STACK["meta_c"], max_iter=1000),
        cv=variant.STACK["cv"], n_jobs=1,
    )


def make_mono(signs):
    cfg = variant.MONO
    constraints = "(" + ",".join(str(int(s)) for s in signs) + ")"
    return VotingClassifier(
        [(f"x{i}", xgb.XGBClassifier(
            n_estimators=cfg["n"], learning_rate=cfg["lr"], max_depth=cfg["depth"],
            min_child_weight=cfg["min_child_weight"], subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample"], reg_lambda=cfg["reg_lambda"],
            gamma=cfg["gamma"], tree_method="hist", monotone_constraints=constraints,
            n_jobs=NJ, random_state=variant.SEED + i, eval_metric="logloss"))
         for i in range(cfg["k"])],
        voting="soft", n_jobs=1,
    )


def make_mlp():
    cfg = variant.MLP
    return VotingClassifier(
        [(f"m{i}", Pipeline([
            ("s", StandardScaler()),
            ("p", PCA(cfg["pca"], random_state=variant.SEED)),
            ("m", MLPClassifier(cfg["hidden"], alpha=cfg["alpha"], max_iter=cfg["max_iter"],
                                early_stopping=True, validation_fraction=0.15,
                                n_iter_no_change=15, random_state=variant.SEED + i)),
        ])) for i in range(cfg["k"])],
        voting="soft", n_jobs=1,
    )


def make_drse():
    return DRSE(**variant.DRSE)


def fit_components(PH, V2, UN, y, signs, rows):
    """Fit all four components on `rows`. Returns the fitted RocketEnsemble parts."""
    return {
        "stack": make_stack().fit(PH[rows], y[rows]),
        "mono": make_mono(signs).fit(PH[rows], y[rows]),
        "mlp": make_mlp().fit(UN[rows], y[rows]),
        "drse": make_drse().fit(V2[rows], y[rows]),
    }


def predict_components(models, PH, V2, UN, rows):
    return {
        "stack": models["stack"].predict_proba(PH[rows])[:, 1],
        "mono": models["mono"].predict_proba(PH[rows])[:, 1],
        "mlp": models["mlp"].predict_proba(UN[rows])[:, 1],
        "drse": models["drse"].predict_proba(V2[rows])[:, 1],
    }


def mine_monotone_signs(PH, y, dates, unique_dates):
    """+1/-1 for phasberg columns whose per-date Spearman sign is stable, else 0."""
    signs = []
    for j in range(PH.shape[1]):
        rhos = []
        for d in unique_dates:
            m = dates == d
            if m.sum() < 8 or len(set(y[m])) < 2:
                continue
            rho = spearmanr(PH[m, j], y[m]).correlation
            if not np.isnan(rho):
                rhos.append(rho)
        if (len(rhos) >= variant.MONO_MIN_DATES
                and abs(np.mean(rhos)) >= variant.MONO_MIN_RHO
                and (np.sign(rhos) == np.sign(np.mean(rhos))).mean() >= variant.MONO_MIN_AGREE):
            signs.append(int(np.sign(np.mean(rhos))))
        else:
            signs.append(0)
    return signs


def select_weights(oof_parts, y_oof):
    """Walk-forward-select the blend weights from variant.W_GRID.

    Every candidate is scored on the SAME held-out component predictions, so this
    compares blends rather than reruns training. The prior (grid[0]) keeps its seat
    unless a rival clears it by W_SELECT_MARGIN.

    Returns (weights, reward, prior_reward). The winner's reward is a max over the
    grid on this pool and is therefore optimistically biased; prior_reward is the
    prior's score on the same pool with no selection applied to it, which is the more
    conservative number. Both are recorded in meta.json.
    """
    prior = variant.W_GRID[0]
    scored = []
    for cand in variant.W_GRID:
        r, _ = reward(blend(oof_parts, cand), y_oof)
        scored.append((float(r), cand))
    prior_reward = scored[0][0]
    best_reward, best = max(scored, key=lambda item: item[0])

    for r, cand in scored:
        tag = "prior" if cand is prior else "     "
        print(f"    {tag} {{{', '.join(f'{k}:{cand[k]:.2f}' for k in PARTS)}}} reward={r:.4f}",
              flush=True)

    if best is not prior and best_reward > prior_reward + variant.W_SELECT_MARGIN:
        print(f"  weights: prior {prior_reward:.4f} -> selected {best_reward:.4f} "
              f"(+{best_reward - prior_reward:.4f} > margin {variant.W_SELECT_MARGIN})", flush=True)
        return dict(best), best_reward, prior_reward
    print(f"  weights: keeping prior ({prior_reward:.4f}); best rival {best_reward:.4f} "
          f"did not clear the {variant.W_SELECT_MARGIN} margin", flush=True)
    return dict(prior), prior_reward, prior_reward


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    examples = load_examples()
    if not examples:
        raise SystemExit(
            "no benchmark data in model/data_cache — run `python3 autopilot.py` "
            "(or its REFRESH step) first"
        )
    chunks = [sanitize(e.hands) for e in examples]
    y = np.array([e.label for e in examples])
    dates = np.array([e.source_date for e in examples])
    unique_dates = sorted(set(dates))

    PH, cols_ph = mat(chunks, phasberg_dict)
    V2, cols_v2 = mat(chunks, v2_dict)
    UN = np.hstack([V2, PH])

    print(f"rocket-{variant.SLUG} | {len(y)} chunks | ph{PH.shape[1]} v2{V2.shape[1]} "
          f"un{UN.shape[1]} | {len(unique_dates)} dates ({time.time() - t0:.0f}s)",
          flush=True)

    # --- walk-forward: train on the past, predict the next unseen date ------ #
    #
    # The monotone signs are re-mined inside every fold from that fold's PAST dates
    # only. dragon-0 mined them once over the whole dataset and reused them in each
    # fold, which quietly leaked: the held-out date's labels helped decide the
    # constraints baked into the `mono` component that was then scored on that very
    # date. It inflates cv_reward — and cv_reward is not decoration here. It gates
    # promotion in autopilot.py, it selects the blend weights below, and it is
    # published in the manifest as a performance claim. A number carrying leakage
    # would make all three dishonest.
    oof_parts = {name: np.full(len(y), np.nan) for name in PARTS}
    for test_date in unique_dates[-WF:]:
        train_rows = dates < test_date
        test_rows = dates == test_date
        if train_rows.sum() < 60 or len(set(y[train_rows])) < 2:
            continue
        past_dates = [d for d in unique_dates if d < test_date]
        fold_signs = mine_monotone_signs(PH[train_rows], y[train_rows],
                                         dates[train_rows], past_dates)
        models = fit_components(PH, V2, UN, y, fold_signs, train_rows)
        preds = predict_components(models, PH, V2, UN, test_rows)
        for name in PARTS:
            oof_parts[name][test_rows] = preds[name]
        print(f"  wf {test_date} | {sum(1 for s in fold_signs if s)} monotone "
              f"(mined on {len(past_dates)} past dates) ({time.time() - t0:.0f}s)", flush=True)

    covered = ~np.isnan(oof_parts["stack"])
    if covered.sum() < 20 or len(set(y[covered])) < 2:
        raise SystemExit("walk-forward produced too little held-out data to fit a blend")
    pooled = {name: oof_parts[name][covered] for name in PARTS}
    y_oof = y[covered]

    weights, _, prior_reward = select_weights(pooled, y_oof)
    oof_scores = blend(pooled, weights)
    cv_ap = float(average_precision_score(y_oof, oof_scores))
    cv_reward, res = reward(oof_scores, y_oof)
    print(f"WALK-FORWARD[{WF}d, n={covered.sum()}]: cv_ap={cv_ap:.4f} reward={cv_reward:.4f} "
          f"recall@fpr={res['bot_recall']:.3f} fpr={res['fpr']:.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    deploy_threshold = fpr_target_threshold(oof_scores[y_oof == 0], TARGET_FPR)

    # --- final fit on everything ------------------------------------------- #
    # The served model may use every date, including the walk-forward ones: nothing is
    # measured on it. Only the cv_* numbers above have to be leak-free.
    all_rows = np.ones(len(y), dtype=bool)
    signs = mine_monotone_signs(PH, y, dates, unique_dates)
    print(f"final fit | {sum(1 for s in signs if s)} monotone constraints "
          f"over all {len(unique_dates)} dates ({time.time() - t0:.0f}s)", flush=True)
    models = fit_components(PH, V2, UN, y, signs, all_rows)
    ens = RocketEnsemble(models["stack"], models["mono"], models["mlp"], models["drse"],
                         cols_ph, cols_v2, weights=weights)
    with open(ART / "model.pkl", "wb") as fh:
        pickle.dump(ens, fh)

    meta = {
        "variant": variant.SLUG,
        "variant_family": variant.FAMILY,
        "fusion": RocketEnsemble.fusion,
        "model_version": variant.VERSION,
        "feature_version": "rocket_phasberg+union+v2",
        "trained_on": "sanitized",
        "model": (f"rocket-{variant.SLUG}: {RocketEnsemble.fusion} fusion("
                  + " + ".join(f"{k} {weights[k]:.2f}" for k in PARTS) + ") [walk-forward]"),
        "weights": weights,
        "weights_prior": variant.W_PRIOR,
        "weights_selected_by": f"walk-forward over {WF} held-out date(s), margin {variant.W_SELECT_MARGIN}",
        "weights_candidates": len(variant.W_GRID),
        # Say plainly what cv_reward is, because the manifest publishes it as a claim:
        # the weights were CHOSEN on this same walk-forward pool, so the number is a max
        # over the candidate grid and is optimistically biased. cv_reward_prior is the
        # untouched prior's score on the same pool — no selection happened against it, so
        # it is the more conservative read. The two are equal whenever the prior held.
        "cv_reward_prior": float(prior_reward),
        "cv_reward_note": (
            f"max over {len(variant.W_GRID)} candidate weightings evaluated on this same "
            f"walk-forward pool; optimistically biased by that selection. Monotone "
            f"constraints are re-mined per fold from past dates only (no label leakage). "
            f"cv_reward_prior is the unselected prior's score on the same pool."
        ),
        "deploy_threshold": float(deploy_threshold),
        "target_fpr": TARGET_FPR,
        "cv_ap": cv_ap,
        "cv_reward": float(cv_reward),
        "cv_recall": float(res["bot_recall"]),
        "cv_fpr": float(res["fpr"]),
        "validation": "walk-forward (train past -> test next unseen date)",
        "reward_formula": "0.75*AP + 0.25*recall@fpr<=0.05 (official 2026-06-26)",
        "n_train": int(len(y)),
        "n_walkforward": int(covered.sum()),
        "n_features_ph": int(PH.shape[1]),
        "n_features_un": int(UN.shape[1]),
        "n_monotone": int(sum(1 for s in signs if s)),
        "n_dates": len(unique_dates),
        "benchmark_releases": unique_dates,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(ART / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    # Baseline for the live-vs-benchmark drift report (autopilot.py + live_probe.py).
    with open(ART / "drift_baseline.json", "w") as fh:
        json.dump({
            "cols_v2": cols_v2,
            "mean": V2.mean(axis=0).tolist(),
            "std": (V2.std(axis=0) + 1e-9).tolist(),
            "n": int(len(y)),
            "trained_at": meta["trained_at"],
        }, fh)

    print(f"saved rocket-{variant.SLUG} -> model.pkl + meta.json | cv_ap={cv_ap:.4f} "
          f"cv_reward={cv_reward:.4f} | weights={weights} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
