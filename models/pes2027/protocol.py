"""Freeze gates. A runner that scores anything new calls require_frozen() first.

The protocol counts as frozen when the YAML is tracked, unmodified in the working tree,
and present in a commit. The returned record (file hash, commit, commit time) is written
into every output so a result can be tied to the protocol it ran under.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from models.pes2027.common import PROTOCOL_DIR, ROOT, file_sha256

PROTOCOL_FILE = PROTOCOL_DIR / "protocol_v1.yaml"


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def require_committed(path: Path) -> dict:
    rel = str(Path(path).resolve().relative_to(ROOT))
    if not Path(path).exists():
        raise SystemExit(f"{rel} does not exist")
    if not _git("ls-files", "--", rel):
        raise SystemExit(f"{rel} is not tracked by git; commit it before running")
    if _git("status", "--porcelain", "--", rel):
        raise SystemExit(f"{rel} has uncommitted changes; commit it before running")
    commit = _git("log", "-1", "--format=%H", "--", rel)
    when = _git("log", "-1", "--format=%cI", "--", rel)
    return {"path": rel, "sha256": file_sha256(path), "commit": commit, "committed": when}


def require_frozen() -> dict:
    return require_committed(PROTOCOL_FILE)
