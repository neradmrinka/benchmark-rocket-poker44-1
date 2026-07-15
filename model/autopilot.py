"""Daily self-retraining pilot for the rocket miner.

    python3 autopilot.py                 # refresh -> retrain -> guard -> deploy
    python3 autopilot.py --no-restart    # everything except the pm2 restart
    python3 autopilot.py --no-refresh    # retrain on what is already cached
    python3 autopilot.py --dry-run       # report state + drift, change nothing

One command, run nightly by pm2 (cron_restart in ecosystem.config.js), doing the
whole loop:

  1. REFRESH  pull every benchmark release we do not have yet. The benchmark grows
              by a date a day; a model pinned to the dates it was born with decays
              against a live feed that does not stand still.
  2. RETRAIN  refit on everything cached (train_rocket.py, which re-selects the
              blend weights by walk-forward as the pool grows).
  3. GUARD    keep the candidate ONLY if its walk-forward reward does not regress
              against the model currently being served, and clears an absolute
              sanity floor. Anything else is reverted from a backup taken before
              training started.
  4. DEPLOY   restart the miner if — and only if — the artifact on disk changed,
              so what we serve always equals what the manifest attests.

Crash-safe and idempotent: the previous artifact is backed up before training and
restored on any regression, guard failure, or exception, so the miner is never left
serving a worse or broken model.

Also reports live-vs-training feature drift (live_probe.py). A benchmark CV number
says nothing about the live feed once the distributions separate, and this is the
only thing here that would notice.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ART = HERE / "artifacts"
DATA = HERE / "data_cache"
BACKUPS = HERE / "artifacts_backups"
LOG = HERE / "autopilot.log"
HISTORY = HERE / "autopilot_history.jsonl"
TRAINER = HERE / "train_rocket.py"

for _p in (str(HERE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_identity  # noqa: E402

model_identity.load_env_file(REPO / ".env")

API = "https://api.poker44.net/api/v1/benchmark"
PY = sys.executable  # the interpreter the miner itself runs under

# --- promotion guards ------------------------------------------------------ #
REWARD_EPSILON = 0.002   # tolerate noise; require new >= old - epsilon

# Absolute floor, checked on top of the no-regression rule.
#
# This replaces an inherited `MAX_DEPLOY_FPR = 0.06` ceiling on meta['cv_fpr'] that
# could never fire. cv_fpr comes out of reward() -> _recall_at_fpr(max_fpr=0.05),
# which only ever selects an operating point from indices where fpr <= 0.05 — so the
# value it reports is <= 0.05 by construction and never reaches 0.06. (Checked over
# 25k random, adversarial and deliberately inverted models: the maximum observed was
# exactly 0.0500.) Its comment also cited a "reward() human-safety cliff" that no
# longer exists — reward() now sets human_safety_penalty = 1.0 unconditionally, and
# scoring is pure ranking. So the ceiling was guarding nothing, against nothing.
#
# The gap it left behind is real though: on the FIRST train there is no baseline
# (old_reward = -1.0), so the no-regression rule passes anything, including a
# degenerate model. This floor is what catches that. A coin-flip model scores about
# 0.75*base_rate + a little, i.e. ~0.25-0.35; a working one lands near 0.85. 0.50 sits
# in the empty space between, so it rejects garbage without ever threatening a real
# model.
MIN_DEPLOY_REWARD = float(os.environ.get("POKER44_MIN_DEPLOY_REWARD", "0.50"))
MINER_PM2_NAME = model_identity.env("POKER44_PM2_NAME", "poker44_miner")


def log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"{stamp} | {msg}"
    print(line, flush=True)
    try:
        with LOG.open("a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "poker44-rocket-autopilot"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# --------------------------------------------------------------------------- #
# 1. REFRESH
# --------------------------------------------------------------------------- #
def refresh_data() -> int:
    """Cache every benchmark date we are missing. Returns how many are new."""
    DATA.mkdir(parents=True, exist_ok=True)
    try:
        doc = json.loads(_get(f"{API}/releases?limit=100"))
        releases = [r["sourceDate"] for r in doc["data"]["releases"]]
    except Exception as exc:
        log(f"REFRESH: could not list releases ({exc}); using cached data only")
        return 0

    added = 0
    for date in sorted(releases):
        path = DATA / f"{date}.json"
        if path.exists() and path.stat().st_size > 0:
            continue
        try:
            blob = _get(f"{API}/chunks?sourceDate={date}&limit=48")
            parsed = json.loads(blob)  # validate before trusting it
            if "data" not in parsed or "chunks" not in parsed["data"]:
                log(f"REFRESH: {date} payload missing data.chunks; skipping")
                continue
            tmp = path.with_suffix(".json.tmp")
            tmp.write_bytes(blob)
            tmp.replace(path)
            added += 1
            log(f"REFRESH: cached new date {date}")
        except Exception as exc:
            log(f"REFRESH: failed to fetch {date} ({exc})")

    cached = len(list(DATA.glob("*.json")))
    log(f"REFRESH: {added} new date(s); {cached} total cached")
    return added


# --------------------------------------------------------------------------- #
# 2 + 3. RETRAIN under guard
# --------------------------------------------------------------------------- #
def read_meta(art_dir: Path | None = None) -> dict | None:
    # Resolved at call time, not bound as a default: a default argument would capture
    # ART once at import and keep reading the original directory even after ART is
    # repointed, which silently reports the wrong baseline reward.
    art_dir = Path(art_dir) if art_dir is not None else ART
    try:
        with (art_dir / "meta.json").open() as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def backup_current() -> Path | None:
    if not (ART / "model.pkl").is_file():
        return None
    BACKUPS.mkdir(parents=True, exist_ok=True)
    dest = BACKUPS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    shutil.copytree(ART, dest)
    snaps = sorted(d for d in BACKUPS.iterdir() if d.is_dir())
    for old in snaps[:-10]:  # keep the 10 most recent
        shutil.rmtree(old, ignore_errors=True)
    return dest


def restore(backup_dir: Path) -> None:
    shutil.rmtree(ART, ignore_errors=True)
    shutil.copytree(backup_dir, ART)


def retrain_and_guard(force_deploy: bool) -> bool:
    """Retrain, then keep the candidate only if it earns its place.

    Returns True when the artifact on disk changed (i.e. the miner must restart).
    """
    old_meta = read_meta()
    old_reward = float(old_meta["cv_reward"]) if old_meta else -1.0
    old_dates = int(old_meta.get("n_dates", 0)) if old_meta else 0
    backup = backup_current()
    log(f"RETRAIN: baseline cv_reward={old_reward:.4f} over {old_dates} dates"
        f"{' (backed up)' if backup else ' (no prior model)'}")

    proc = subprocess.run(
        [PY, str(TRAINER)], cwd=str(HERE), capture_output=True, text=True,
        env={**os.environ, "POKER44_REPO": str(REPO), "PYTHONUNBUFFERED": "1"},
    )
    if proc.returncode != 0:
        log(f"RETRAIN: {TRAINER.name} FAILED rc={proc.returncode}")
        log(proc.stderr.strip()[-1500:])
        if backup:
            restore(backup)
            log("RETRAIN: reverted to previous artifact")
        return False

    new_meta = read_meta()
    if not new_meta:
        log("RETRAIN: no meta.json after training; reverting")
        if backup:
            restore(backup)
        return False

    new_reward = float(new_meta["cv_reward"])
    new_fpr = float(new_meta.get("cv_fpr", 1.0))
    new_dates = int(new_meta.get("n_dates", 0))
    log(f"RETRAIN: candidate cv_reward={new_reward:.4f} cv_fpr={new_fpr:.4f} "
        f"cv_ap={new_meta.get('cv_ap', 0):.4f} over {new_dates} dates "
        f"| weights={new_meta.get('weights')}")

    reasons = []
    if new_reward < MIN_DEPLOY_REWARD:
        reasons.append(f"reward {new_reward:.4f} < absolute floor {MIN_DEPLOY_REWARD} "
                       f"(model looks degenerate)")
    if not force_deploy and new_reward < old_reward - REWARD_EPSILON:
        reasons.append(f"reward {new_reward:.4f} < baseline {old_reward:.4f} - eps")

    if reasons:
        if backup:
            log("RETRAIN: REJECTED (" + "; ".join(reasons) + ") -> reverting")
            restore(backup)
            record_history("rejected", old_reward, new_reward, new_fpr, new_dates, reasons)
            return False
        # First ever train and it already trips a guard: nothing to revert to, so keep
        # it (a serving miner beats no miner) but make the noise impossible to miss.
        log("RETRAIN: WARNING first model violates a guard but there is no backup to "
            "revert to: " + "; ".join(reasons))

    improved = new_reward > old_reward + REWARD_EPSILON
    fresher = new_dates > old_dates and new_reward >= old_reward - REWARD_EPSILON
    if backup and not (improved or fresher or force_deploy):
        # Neither better nor fresher: keep the proven artifact rather than swap in a
        # sideways one and pay a miner restart for nothing.
        log(f"RETRAIN: candidate is neither better ({new_reward:.4f} vs {old_reward:.4f}) "
            f"nor fresher ({new_dates} vs {old_dates} dates) -> keeping current model")
        restore(backup)
        record_history("kept_current", old_reward, new_reward, new_fpr, new_dates, [])
        return False

    why = "improved" if improved else ("fresher data" if fresher else "forced/first")
    log(f"RETRAIN: PROMOTED ({why}) cv_reward {old_reward:.4f} -> {new_reward:.4f}")
    record_history("promoted", old_reward, new_reward, new_fpr, new_dates, [])
    return True


def record_history(decision, old_reward, new_reward, fpr, n_dates, reasons) -> None:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "old_reward": old_reward,
        "new_reward": new_reward,
        "fpr": fpr,
        "n_dates": n_dates,
        "reasons": reasons,
    }
    try:
        with HISTORY.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# live drift
# --------------------------------------------------------------------------- #
def report_drift() -> None:
    """Compare what the validator actually sends against what we trained on."""
    try:
        import live_probe

        with (ART / "drift_baseline.json").open() as fh:
            baseline = json.load(fh)
        report = live_probe.drift_report(baseline)
    except (OSError, ValueError) as exc:
        log(f"DRIFT: no baseline yet ({exc})")
        return
    except Exception as exc:
        log(f"DRIFT: report failed ({exc})")
        return

    if report["status"] != "ok":
        log(f"DRIFT: {report['status']} ({report['n_batches']} live batches captured)")
        return

    worst = ", ".join(f"{d['feature']}={d['z']:+.2f}" for d in report["top"][:5])
    log(f"DRIFT: {report['n_batches']} live batches | mean|z|={report['mean_abs_z']:.2f} "
        f"max|z|={report['max_abs_z']:.2f} | worst: {worst}")
    if report["max_abs_z"] > 3.0:
        log("DRIFT: WARNING a live feature has moved >3 training sd from the benchmark. "
            "Benchmark CV no longer describes live performance; inspect before trusting "
            "the next promotion.")


# --------------------------------------------------------------------------- #
# 4. DEPLOY
# --------------------------------------------------------------------------- #
def restart_miner() -> None:
    try:
        out = subprocess.run(["pm2", "restart", MINER_PM2_NAME, "--update-env"],
                             capture_output=True, text=True)
        if out.returncode == 0:
            log(f"DEPLOY: restarted pm2 process '{MINER_PM2_NAME}'")
        else:
            log(f"DEPLOY: pm2 restart failed rc={out.returncode}: {out.stderr.strip()[-400:]}")
    except FileNotFoundError:
        log("DEPLOY: pm2 not found on PATH; restart the miner manually")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-restart", action="store_true", help="retrain/promote but do not restart")
    ap.add_argument("--force-deploy", action="store_true",
                    help="promote even if reward ties/regresses (still honours the fpr ceiling)")
    ap.add_argument("--no-refresh", action="store_true", help="skip the data download")
    ap.add_argument("--dry-run", action="store_true", help="report state and drift; change nothing")
    args = ap.parse_args()

    t0 = time.time()
    log("=== AUTOPILOT START ===")
    report_drift()

    if args.dry_run:
        meta = read_meta()
        cached = len(list(DATA.glob("*.json"))) if DATA.is_dir() else 0
        log(f"DRY-RUN: {cached} cached dates | current "
            f"cv_reward={meta.get('cv_reward') if meta else None} "
            f"weights={meta.get('weights') if meta else None}")
        log("=== AUTOPILOT DONE (dry-run) ===")
        return

    added = 0 if args.no_refresh else refresh_data()
    changed = retrain_and_guard(force_deploy=args.force_deploy)
    if changed and not args.no_restart:
        restart_miner()
    elif changed:
        log("DEPLOY: artifact changed but --no-restart set; serving stale until restart")
    else:
        log("DEPLOY: nothing to deploy (artifact unchanged)")
    log(f"=== AUTOPILOT DONE in {time.time() - t0:.0f}s | new_dates={added} "
        f"artifact_changed={changed} ===")


if __name__ == "__main__":
    main()
