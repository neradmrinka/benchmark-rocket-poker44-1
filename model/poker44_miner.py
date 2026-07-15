"""rocket-r1 miner entrypoint — serves the balanced rank-fusion model on subnet 126.

Run (pm2 does this for you from ecosystem.config.js, which reads .env):

    python3 model/poker44_miner.py --netuid 126 \
        --wallet.name <coldkey> --wallet.hotkey <hotkey> \
        --subtensor.network finney --axon.port <port> \
        --blacklist.force_validator_permit

Identity is not decoration here. A validator that cannot verify who we are marks us
`opaque` and applies a transparency penalty, so the manifest is built by
model_identity.py, which resolves a real git commit, publishes a digest anyone can
recompute from the public repo, and refuses to stay quiet when either is broken.
"""

# NOTE: do NOT `from __future__ import annotations` here. bittensor's axon.attach
# introspects the real type of forward()'s `synapse` parameter via issubclass();
# stringized (PEP 563) annotations break that with "issubclass() arg 1 must be a
# class". The reference miner omits the future-import for the same reason.

import os
import sys
import time
from pathlib import Path
from typing import Tuple

MODEL_DIR = Path(__file__).resolve().parent
REPO_DIR = Path(os.environ.get("POKER44_REPO") or MODEL_DIR.parent).resolve()
for _p in (str(MODEL_DIR), str(REPO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bittensor as bt

import model_identity
import variant
from poker44.base.miner import BaseMinerNeuron
from poker44.validator.synapse import DetectionSynapse

# pm2 injects .env via ecosystem.config.js; this makes a bare `python3 poker44_miner.py`
# behave identically instead of silently serving with no repo/commit identity.
model_identity.load_env_file(REPO_DIR / ".env")

from infer import get_model

# Published sources that define this model, repo-relative. These are what
# implementation_sha256 covers, so a third party can clone the repo at repo_commit
# and recompute the same digest. The trained weights are NOT here — they stay
# private and are attested by artifact_sha256 instead.
SOURCE_FILES = [
    "model/poker44_miner.py",
    "model/infer.py",
    "model/rocket_ensemble.py",
    "model/variant.py",
    "model/train_rocket.py",
    "model/model_identity.py",
    "model/live_probe.py",
    "model/d0_features.py",
    "model/d0_drse.py",
    "model/features_v2.py",
    "model/dataset.py",
    "model/reward_fn.py",
    "model/poker44_ml/features.py",
]
ARTIFACT = MODEL_DIR / "artifacts" / "model.pkl"


class RocketMiner(BaseMinerNeuron):
    """Serves rocket-r1: weighted rank fusion over four decorrelated components."""

    def __init__(self, config=None):
        super().__init__(config=config)
        self.poker_model = get_model()
        meta = self.poker_model.meta

        weights = meta.get("weights", variant.W_PRIOR)
        self.model_manifest = model_identity.build_manifest(
            repo_root=REPO_DIR,
            source_files=SOURCE_FILES,
            artifact_path=ARTIFACT,
            defaults={
                "model_name": model_identity.env("POKER44_MODEL_NAME", f"rocket-{variant.SLUG}"),
                "model_version": variant.VERSION,
                "framework": variant.FRAMEWORK,
                "license": "MIT",
                "repo_url": model_identity.env("POKER44_MODEL_REPO_URL"),
                "open_source": True,
                "inference_mode": "remote",
                "notes": (
                    f"rocket-{variant.SLUG} ({variant.FAMILY}). {variant.SUMMARY} "
                    f"Blend {', '.join(f'{k} {weights[k]:.2f}' for k in ('stack', 'mono', 'mlp', 'drse'))} "
                    f"selected by walk-forward. Walk-forward AP={meta['cv_ap']:.4f} "
                    f"reward={meta['cv_reward']:.4f} over {meta['n_dates']} benchmark dates. "
                    f"Source is published; trained weights are private and attested by "
                    f"artifact_sha256."
                ),
                "training_data_statement": (
                    "Trained only on the PUBLIC Poker44 benchmark "
                    "(api.poker44.net/api/v1/benchmark), retrained daily as new dates are "
                    "released. Every hand is projected through the validator's own "
                    "prepare_hand_for_miner() before featurisation, so training matches the "
                    "sanitized view that is served. No validator-only data is used."
                ),
                "training_data_sources": ["poker44-public-benchmark"],
                "private_data_attestation": (
                    "This model does not train on validator-only evaluation data. Live "
                    "validator payloads are never retained; only anonymous aggregate feature "
                    "means are logged locally for drift monitoring (model/live_probe.py)."
                ),
                "data_attestation": (
                    "Features read only miner-visible behavioural fields: action types, "
                    "sequences, bb-quantised sizes, pot dynamics and seat aliases. No cards, "
                    "board, outcome or identifiers are used."
                ),
            },
        )
        self.manifest_verdict = model_identity.check_manifest(self.model_manifest, REPO_DIR)
        model_identity.log_manifest(self.model_manifest, self.manifest_verdict, bt.logging)
        bt.logging.info(
            f"rocket-{variant.SLUG} ready | fusion={meta.get('fusion')} weights={weights} "
            f"cv_ap={meta['cv_ap']:.4f} cv_reward={meta['cv_reward']:.4f} "
            f"threshold={self.poker_model.threshold:.4f}"
        )
        bt.logging.info(f"Axon created: {self.axon}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        try:
            scores = self.poker_model.score_chunks(chunks)
        except Exception as exc:  # a malformed request must never take the miner down
            bt.logging.warning(f"scoring failed ({exc}); falling back to 0.5")
            scores = [0.5] * len(chunks)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(
            f"Scored {len(chunks)} chunks | bots={sum(synapse.predictions)} "
            f"mean={sum(scores) / max(len(scores), 1):.3f}"
        )
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with RocketMiner() as miner:
        bt.logging.info(f"rocket-{variant.SLUG} miner running...")
        while True:
            try:
                bt.logging.info(
                    f"UID {miner.uid} | incentive {miner.metagraph.I[miner.uid]:.6f}")
            except Exception:
                pass
            time.sleep(5 * 60)
