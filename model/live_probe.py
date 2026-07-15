"""Live synapse probe — is the model still being asked the question it was trained on?

We train on the PUBLIC benchmark pushed through the validator's own
prepare_hand_for_miner(). We are *scored* on live chunks the platform's eval API
produces. Those two only stay the same distribution for as long as the platform's
data and its sanitizer stay put — and when that stops being true nothing announces
it. The score just quietly decays, and a benchmark-only CV number keeps looking
healthy while it happens.

So: record the per-column mean of the v2 feature matrix for a sample of live
batches. autopilot.py diffs that against the training baseline written by
train_rocket.py (artifacts/drift_baseline.json) and reports which features moved,
in units of training standard deviations.

What is stored is a ~250-float summary per sampled batch — no hands, no payloads,
no identifiers. Validator evaluation material is never retained; that matters both
for the private_data_attestation we publish and for not being the miner that
hoards eval data.

Cheap by construction: one column-mean over a matrix that is already in memory, at
POKER44_PROBE_RATE (default: every 20th batch), and every entry point swallows its
own exceptions — a probe must never be able to damage a response.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

CAPTURE_DIR = Path(__file__).resolve().parent / "live_capture"
LOG_PATH = CAPTURE_DIR / "live_v2.jsonl"

PROBE_RATE = max(1, int(os.environ.get("POKER44_PROBE_RATE", "20")))
MAX_RECORDS = max(50, int(os.environ.get("POKER44_PROBE_MAX_RECORDS", "2000")))

# Trim only once the file is meaningfully over budget, and decide that from a stat()
# rather than by reading it. Checking the line count meant read_text()ing the whole log
# on every sampled batch, and once it sat at the cap that turned into a multi-megabyte
# read-and-rewrite inside every 20th scored response — synchronous latency on a reply
# the validator is timing. Now: one stat() per sampled batch, and a rewrite roughly
# once per (HIGH_WATER - MAX_RECORDS) records, amortised to nothing.
_BYTES_PER_RECORD = 4096          # generous upper bound for ~250 rounded floats
HIGH_WATER_BYTES = int(MAX_RECORDS * 1.25) * _BYTES_PER_RECORD

_counter = 0


def record_batch(v2_matrix, *, n_chunks: int) -> None:
    """Append a column-mean summary of one live v2 batch. Sampled; never raises."""
    global _counter
    try:
        _counter += 1
        if (_counter - 1) % PROBE_RATE:
            return
        import numpy as np

        matrix = np.asarray(v2_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.size == 0:
            return

        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_chunks": int(n_chunks),
            "mean": [round(float(v), 6) for v in matrix.mean(axis=0)],
        }
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        _trim()
    except Exception:
        return


def _trim() -> None:
    """Keep the log bounded; it runs forever on a miner box.

    The stat() guard is the point: without it this reads and rewrites the entire log
    on every sampled batch once the cap is reached.
    """
    try:
        if LOG_PATH.stat().st_size <= HIGH_WATER_BYTES:
            return
        lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
        if len(lines) <= MAX_RECORDS:
            return
        tmp = LOG_PATH.with_suffix(".tmp")
        tmp.write_text("\n".join(lines[-MAX_RECORDS:]) + "\n", encoding="utf-8")
        tmp.replace(LOG_PATH)
    except OSError:
        return


def load_records(limit: int = 200) -> List[Dict[str, Any]]:
    try:
        lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: List[Dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def drift_report(baseline: Dict[str, Any], limit: int = 200, top: int = 8) -> Dict[str, Any]:
    """Standardised live-vs-training drift per v2 column.

    z_j = (mean_live_j - mean_train_j) / std_train_j — "how many training standard
    deviations has this feature moved". Returns the worst offenders. A handful of
    large |z| means the served distribution has walked away from the trained one and
    the benchmark CV number is no longer describing live performance.
    """
    records = load_records(limit)
    if not records or not baseline:
        return {"status": "no_data", "n_batches": len(records), "top": []}

    import numpy as np

    cols = baseline.get("cols_v2") or []
    mu = np.asarray(baseline.get("mean") or [], dtype=float)
    sd = np.asarray(baseline.get("std") or [], dtype=float)
    live = np.asarray([r["mean"] for r in records if len(r.get("mean", [])) == len(cols)],
                      dtype=float)
    if live.size == 0 or len(cols) != mu.size or mu.size != sd.size:
        return {"status": "shape_mismatch", "n_batches": len(records), "top": []}

    z = (live.mean(axis=0) - mu) / np.maximum(sd, 1e-9)
    order = np.argsort(-np.abs(z))[:top]
    return {
        "status": "ok",
        "n_batches": int(live.shape[0]),
        "max_abs_z": float(np.abs(z).max()),
        "mean_abs_z": float(np.abs(z).mean()),
        "top": [{"feature": cols[i], "z": round(float(z[i]), 3)} for i in order],
    }
