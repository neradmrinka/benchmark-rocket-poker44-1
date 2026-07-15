// pm2 process definitions for the rocket-r1 miner.
//
// SECURITY: no wallet, hotkey, port or repo is hard-coded here — this file is
// committed. Everything operational comes from `.env` (gitignored). Copy
// `.env.example` -> `.env` and fill it in before starting.
//
// Paths are derived from this file's own location, so the project can live
// anywhere. (The upstream config hard-coded /root/my_pocker/pocker_d0 and broke
// the moment it was cloned somewhere else.)
//
//   pm2 start model/ecosystem.config.js
//   pm2 logs <POKER44_PM2_NAME>

const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

const MODEL = __dirname;
const REPO = path.resolve(MODEL, "..");

function loadEnv(p) {
  const out = {};
  try {
    for (const raw of fs.readFileSync(p, "utf8").split("\n")) {
      const line = raw.trim();
      if (!line || line.startsWith("#")) continue;
      const i = line.indexOf("=");
      if (i > 0) {
        out[line.slice(0, i).trim()] = line
          .slice(i + 1)
          .trim()
          .replace(/^["']|["']$/g, "");
      }
    }
  } catch (e) {
    /* no .env yet — the guards below will say so */
  }
  return out;
}

// Drop empty values so a blank `KEY=` in .env falls through to a default here
// instead of overriding it with "". POKER44_MODEL_REPO_COMMIT= ships empty, and an
// empty commit in the manifest is exactly what gets a miner marked opaque.
const FILE_ENV = loadEnv(path.join(REPO, ".env"));
const E = {};
for (const [k, v] of Object.entries({ ...FILE_ENV, ...process.env })) {
  if (v !== undefined && v !== null && String(v).trim() !== "") E[k] = String(v).trim();
}

const WALLET = E.POKER44_WALLET_NAME;
const HOTKEY = E.POKER44_WALLET_HOTKEY;
const NETUID = E.POKER44_NETUID || "126";
const PORT = E.POKER44_AXON_PORT || "8091";
const REPO_URL = E.POKER44_MODEL_REPO_URL || "";
const MODEL_NAME = E.POKER44_MODEL_NAME || "rocket-r1";
const PM2_NAME = E.POKER44_PM2_NAME || "poker44_miner_r1";
const PY = E.POKER44_PYTHON || path.join(REPO, "miner_env", "bin", "python");

if (!WALLET || !HOTKEY) {
  throw new Error(
    "rocket-r1: missing POKER44_WALLET_NAME / POKER44_WALLET_HOTKEY — create .env from .env.example"
  );
}
if (!REPO_URL) {
  throw new Error(
    "rocket-r1: missing POKER44_MODEL_REPO_URL — the manifest needs a published repo, " +
      "or the miner is marked opaque"
  );
}

// The manifest must carry a real commit. Prefer an explicit pin from .env; otherwise
// publish the checkout's HEAD. model_identity.py re-resolves this at startup and
// warns if the tree is dirty or the commit was never pushed.
let REPO_COMMIT = E.POKER44_MODEL_REPO_COMMIT || "";
if (!/^[0-9a-f]{7,40}$/.test(REPO_COMMIT)) {
  try {
    REPO_COMMIT = execSync(`git -C ${REPO} rev-parse HEAD`, {
      stdio: ["ignore", "pipe", "ignore"],
    })
      .toString()
      .trim();
  } catch (e) {
    REPO_COMMIT = "";
  }
}
if (!REPO_COMMIT) {
  console.warn(
    "rocket-r1: WARNING no repo commit resolved (not a git checkout, and no " +
      "POKER44_MODEL_REPO_COMMIT in .env) — the miner will report itself OPAQUE."
  );
}

const SHARED = {
  ...E,
  POKER44_REPO: REPO,
  POKER44_MODEL_NAME: MODEL_NAME,
  POKER44_MODEL_REPO_URL: REPO_URL,
  POKER44_MODEL_REPO_COMMIT: REPO_COMMIT,
  POKER44_PM2_NAME: PM2_NAME,
};

module.exports = {
  apps: [
    {
      name: PM2_NAME,
      script: path.join(MODEL, "poker44_miner.py"),
      interpreter: PY,
      cwd: REPO,
      args: [
        "--netuid", NETUID,
        "--wallet.name", WALLET,
        "--wallet.hotkey", HOTKEY,
        "--subtensor.network", "finney",
        "--axon.port", PORT,
        "--logging.debug",
        "--blacklist.force_validator_permit",
      ].join(" "),
      env: SHARED,
      autorestart: true,
      max_restarts: 20,
      min_uptime: "30s",
      restart_delay: 5000,
      kill_timeout: 10000,
    },
    {
      // Nightly: refresh the benchmark, retrain under guard, restart on promotion.
      // 00:16 UTC — a few minutes after the daily benchmark drop.
      name: `${PM2_NAME}_autopilot`,
      script: path.join(MODEL, "autopilot.py"),
      interpreter: PY,
      cwd: MODEL,
      env: SHARED,
      autorestart: false,
      cron_restart: "16 0 * * *",
    },
  ],
};
