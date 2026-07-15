"""rocket-r1 "balanced" — this miner's identity and tuned hyper-parameters.

Everything that separates this miner from its siblings (r2 logit-fusion, r3
consensus-fusion, r4 trimmed-fusion) lives in this file and in rocket_ensemble.py.
The feature views, the data loader and the promotion guards are shared — the
*model* is not.

r1 in one line: the dragon-0 recipe (weighted rank blend of stack / monoXGB /
PCA-MLP / DRSE), with the blend weights fitted by walk-forward instead of
hard-coded, and a prior that leans away from the phasberg-only models.

Why the prior moved off d0's 0.30/0.20/0.30/0.20:
  The validator serves every hand through payload_view.prepare_hand_for_miner,
  which forces button_seat=0, drops the blinds entirely, re-aliases seats and keeps
  only a random 5-8 action window. The phasberg view (poker44_ml/features.py) still
  carries hero/button/blind-derived columns that are constant-zero once sanitized,
  so the two models that see *only* that view (stack, mono) are the ones with the
  least live signal to work with. The v2 view is sanitization-invariant by
  construction, so drse (v2) and mlp (v2+phasberg union) get the weight instead:
  stack+mono 0.50 -> 0.44, mlp+drse 0.50 -> 0.56.

This prior is a hypothesis, not a result. train_rocket.py walk-forward-scores every
candidate in W_GRID — including d0's original point — on held-out dates and only
moves off the prior when the data pays for it by more than W_SELECT_MARGIN.
"""

# --- published identity (mirrored into the manifest) ------------------------ #
SLUG = "r1"
FAMILY = "balanced-rank-fusion"
VERSION = "4.1"
FRAMEWORK = "sklearn-stack+monotone-xgb+pca-mlp+drse/weighted-rank-fusion"
SUMMARY = (
    "Weighted rank fusion of four decorrelated components over three feature views "
    "(phasberg / union / v2). Blend weights selected by walk-forward."
)

# --- blend weights --------------------------------------------------------- #
# Prior = where this variant believes the optimum sits. The grid is searched by
# walk-forward on held-out dates; W_GRID[0] MUST be the prior (train_rocket.py
# falls back to it when nothing beats it convincingly).
W_PRIOR = {"stack": 0.26, "mono": 0.18, "mlp": 0.30, "drse": 0.26}
W_GRID = [
    W_PRIOR,
    {"stack": 0.30, "mono": 0.20, "mlp": 0.30, "drse": 0.20},  # d0's published point
    {"stack": 0.22, "mono": 0.16, "mlp": 0.32, "drse": 0.30},
    {"stack": 0.30, "mono": 0.16, "mlp": 0.28, "drse": 0.26},
    {"stack": 0.24, "mono": 0.22, "mlp": 0.30, "drse": 0.24},
    {"stack": 0.26, "mono": 0.18, "mlp": 0.34, "drse": 0.22},
    {"stack": 0.28, "mono": 0.14, "mlp": 0.28, "drse": 0.30},
]
# Reward gain required before abandoning the prior. The walk-forward pool is only a
# few dates deep, so a hair's-breadth win is noise, not evidence.
W_SELECT_MARGIN = 0.002

# --- component hyper-parameters -------------------------------------------- #
SEED = 1

STACK = dict(
    lgb_n=600, lgb_lr=0.03, lgb_leaves=63,
    xgb_n=500, xgb_lr=0.03, xgb_depth=6,
    cat_n=600, cat_lr=0.03, cat_depth=6,
    et_n=500, et_depth=14,
    rf_n=400, rf_depth=14,
    meta_c=1.0, cv=4,
)
MONO = dict(k=3, n=450, lr=0.035, depth=4, min_child_weight=5,
            subsample=0.8, colsample=0.8, reg_lambda=3.0, gamma=0.5)
MLP = dict(k=3, pca=50, hidden=(64,), alpha=2.0, max_iter=700)
DRSE = dict(n=12, ff=0.70, seed=11)

# --- monotone-constraint mining -------------------------------------------- #
# A phasberg column earns a +1/-1 constraint only if its per-date Spearman sign is
# stable across dates. Steadier than d0's (>=5 dates, 70% agreement, |rho|>=0.05).
MONO_MIN_DATES = 5
MONO_MIN_AGREE = 0.72
MONO_MIN_RHO = 0.05
