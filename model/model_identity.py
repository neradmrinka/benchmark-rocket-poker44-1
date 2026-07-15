"""Verifiable model identity for the Poker44 miner manifest.

A validator marks a miner `opaque` — and applies the transparency/integrity
penalty — unless every field in MIN_REQUIRED_MANIFEST_FIELDS is present *and*
repo_commit matches ^[0-9a-f]{7,40}$ (see
poker44/utils/model_manifest.evaluate_manifest_compliance and
poker44/validator/integrity.evaluate_manifest_suspicion). There are three traps
on the way there, and this module exists to close all three:

1. `.env` ships `POKER44_MODEL_REPO_COMMIT=` — set but EMPTY. os.getenv returns
   "" for that, not the default, so build_local_model_manifest()'s fallback never
   fires and the manifest goes out with repo_commit="" => 'repo_commit_invalid'
   => opaque. resolve_repo_commit() treats empty as unset and falls back to the
   real git HEAD.

2. The shared _sha256_for_files() folds each file's ABSOLUTE path into the digest,
   so implementation_sha256 depends on where the repo happens to sit on disk. Nobody
   who clones the repo can reproduce it => "implementation details cannot be
   verified". implementation_digest() below hashes repo-RELATIVE paths + content,
   so a third party can recompute it from a clean clone at repo_commit.

3. A dirty working tree or an unpushed commit means the code being served is not
   the code at the published commit. audit_repo_state() surfaces both, loudly,
   instead of letting the miner advertise a commit that proves nothing.

The trained weights stay private (see .gitignore); they are attested — not
published — via artifact_sha256, so the identity claim is checkable without
handing out the model.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)

GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
COMMIT_FILE = ".model_commit"


# --------------------------------------------------------------------------- #
# .env loading
# --------------------------------------------------------------------------- #
def load_env_file(path: Path, *, override: bool = False) -> Dict[str, str]:
    """Read a KEY=VALUE .env into os.environ.

    pm2 injects these via ecosystem.config.js, but a manual `python poker44_miner.py`
    would otherwise start with no wallet/port/repo config at all. Existing env vars
    win unless override=True, so the launcher always has the final say.
    """
    loaded: Dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return loaded
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = value
        if override or not os.environ.get(key):
            os.environ[key] = value
    return loaded


def _clean(value: Optional[str]) -> str:
    return (value or "").strip()


def env(name: str, default: str = "") -> str:
    """os.getenv, but a set-but-empty var counts as unset (the .env commit trap)."""
    return _clean(os.environ.get(name)) or default


# --------------------------------------------------------------------------- #
# git identity
# --------------------------------------------------------------------------- #
def _git(repo_root: Path, *args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def looks_like_commit(value: Any) -> bool:
    return bool(GIT_COMMIT_RE.fullmatch(_clean(str(value))))


def resolve_repo_commit(repo_root: Path) -> str:
    """The commit the published manifest points at.

    Precedence: POKER44_MODEL_REPO_COMMIT (if non-empty and well formed) -> the
    repo's git HEAD -> a recorded `.model_commit` file (for tarball deploys with
    no .git). Returns "" if nothing resolves, which the caller must treat as fatal
    for transparency.
    """
    from_env = env("POKER44_MODEL_REPO_COMMIT")
    if looks_like_commit(from_env):
        return from_env.lower()

    head = _git(repo_root, "rev-parse", "HEAD")
    if looks_like_commit(head):
        return str(head).lower()

    try:
        recorded = _clean(Path(repo_root, COMMIT_FILE).read_text(encoding="utf-8"))
        if looks_like_commit(recorded):
            return recorded.lower()
    except OSError:
        pass
    return ""


def audit_repo_state(repo_root: Path, commit: str) -> List[str]:
    """Reasons the published commit may not describe what we actually serve."""
    warnings: List[str] = []
    if not looks_like_commit(commit):
        warnings.append(
            "repo_commit unresolved: set POKER44_MODEL_REPO_COMMIT in .env, or run "
            "the miner from a git checkout so HEAD can be read"
        )
        return warnings

    if _git(repo_root, "rev-parse", "--git-dir") is None:
        warnings.append("not a git checkout: the published commit cannot be cross-checked here")
        return warnings

    # An explicit pin outranks git HEAD by design (it is the escape hatch for deploys
    # with no .git). But a pin that disagrees with the checkout it is running from is
    # almost always a STALE pin — pm2 resolves the commit once at start and keeps
    # handing back that value across restarts, so a `git pull` since then would have
    # us advertising one commit while serving the code of another. Unverifiable, and
    # silent unless someone says so.
    head = _git(repo_root, "rev-parse", "HEAD")
    if looks_like_commit(head) and str(head).lower() != commit.lower():
        warnings.append(
            f"published commit {commit[:10]} != checkout HEAD {str(head)[:10]}: the pin in "
            f"POKER44_MODEL_REPO_COMMIT (or pm2's cached env) is stale. Clear it to publish "
            f"HEAD, or re-pin it to the commit actually being served"
        )

    dirty = _git(repo_root, "status", "--porcelain")
    if dirty:
        n = len([ln for ln in dirty.splitlines() if ln.strip()])
        warnings.append(
            f"working tree is dirty ({n} uncommitted change(s)): the code being served does "
            f"not match commit {commit[:10]}, so implementation_sha256 is not reproducible "
            f"from the published repo"
        )

    on_remote = _git(repo_root, "branch", "-r", "--contains", commit)
    if on_remote is None or not on_remote.strip():
        warnings.append(
            f"commit {commit[:10]} is not on any remote branch: push it before it can be "
            f"verified against the published repo"
        )
    return warnings


# --------------------------------------------------------------------------- #
# content digests
# --------------------------------------------------------------------------- #
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return ""
    return digest.hexdigest()


def implementation_digest(repo_root: Path, rel_files: Iterable[str]) -> str:
    """Path-independent digest over the published sources that define this model.

    Hashes `<repo-relative path>\\0<sha256 of contents>\\n` per file, sorted. Anyone
    can recompute this from a clean clone at repo_commit — which is the whole point
    of publishing it. Missing files are skipped (and reported by missing_sources()).
    """
    lines: List[str] = []
    for rel in sorted(set(rel_files)):
        target = Path(repo_root, rel)
        if not target.is_file():
            continue
        lines.append(f"{rel}\0{file_sha256(target)}")
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def missing_sources(repo_root: Path, rel_files: Iterable[str]) -> List[str]:
    return sorted(rel for rel in set(rel_files) if not Path(repo_root, rel).is_file())


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def build_manifest(
    *,
    repo_root: Path,
    source_files: List[str],
    artifact_path: Path,
    defaults: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build a manifest whose every identity claim can actually be checked.

    `source_files` are repo-relative paths to the PUBLISHED code that defines the
    served model. `artifact_path` is the private weights file — hashed, never shipped.
    """
    repo_root = Path(repo_root).resolve()
    commit = resolve_repo_commit(repo_root)

    present = [rel for rel in source_files if Path(repo_root, rel).is_file()]
    merged = dict(defaults)
    merged["repo_commit"] = commit

    artifact_hash = file_sha256(artifact_path) if Path(artifact_path).is_file() else ""
    if artifact_hash:
        merged["artifact_sha256"] = artifact_hash

    # build_local_model_manifest() reads repo_commit straight from the environment and
    # would otherwise pick up the empty value .env ships, ignoring our default. Feed it
    # the resolved commit, then put the environment back exactly as we found it —
    # publishing a manifest should not leave a pin behind for anything that reads the
    # env after us (autopilot's trainer subprocess inherits this process's environ).
    previous = os.environ.get("POKER44_MODEL_REPO_COMMIT")
    if commit:
        os.environ["POKER44_MODEL_REPO_COMMIT"] = commit
    try:
        manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[Path(repo_root, rel) for rel in present],
            defaults=merged,
        )
    finally:
        if previous is None:
            os.environ.pop("POKER44_MODEL_REPO_COMMIT", None)
        else:
            os.environ["POKER44_MODEL_REPO_COMMIT"] = previous

    # Replace the absolute-path-dependent digest with one a third party can reproduce.
    manifest["implementation_sha256"] = implementation_digest(repo_root, present)
    manifest["implementation_files"] = sorted(present)
    return manifest


def check_manifest(manifest: Mapping[str, Any], repo_root: Path) -> Dict[str, Any]:
    """Full transparency verdict: schema compliance + can-anyone-verify-this checks."""
    compliance = evaluate_manifest_compliance(manifest)
    warnings = audit_repo_state(Path(repo_root), str(manifest.get("repo_commit", "")))
    if not _clean(str(manifest.get("artifact_sha256", ""))):
        warnings.append("artifact_sha256 missing: train the model before serving")
    return {
        "status": compliance["status"],
        "missing_fields": compliance["missing_fields"],
        "policy_violations": compliance["policy_violations"],
        "open_source": compliance["open_source"],
        "warnings": warnings,
        "digest": manifest_digest(manifest),
    }


def log_manifest(manifest: Mapping[str, Any], verdict: Mapping[str, Any], logger) -> None:
    logger.info(
        f"manifest | model={manifest.get('model_name', '')} v{manifest.get('model_version', '')} "
        f"repo={manifest.get('repo_url', '')} commit={str(manifest.get('repo_commit', ''))[:10]} "
        f"open_source={manifest.get('open_source')}"
    )
    logger.info(
        f"manifest | impl_sha256={str(manifest.get('implementation_sha256', ''))[:16]} "
        f"artifact_sha256={str(manifest.get('artifact_sha256', ''))[:16]} "
        f"digest={str(verdict.get('digest', ''))[:16]}"
    )
    if verdict["status"] == "transparent":
        logger.info("manifest | transparency status: TRANSPARENT")
    else:
        logger.error(
            f"manifest | transparency status: OPAQUE — this WILL cost score. "
            f"missing={verdict['missing_fields']} violations={verdict['policy_violations']}"
        )
    for warning in verdict["warnings"]:
        logger.warning(f"manifest | {warning}")
