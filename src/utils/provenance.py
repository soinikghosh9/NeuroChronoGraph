"""Provenance stamping for every result artefact.

F6. No file under `outputs/results/` currently records the code version, the configuration or the
environment that produced it. That has already cost this project real time: results computed under
different `NCG_*` settings sit side by side in the same directory with no way to tell them apart,
and a reviewer asking "which commit produced figure 3" cannot be answered.

Every writer should now go through `write_result`, which attaches a `provenance` block:

    {
      "provenance": {
        "git_commit": "fd56204...", "git_dirty": true,
        "script": "119_cost_decomposition.py",
        "written_utc": "2026-08-04T14:22:10Z",
        "env": {"NCG_NORM": "channel", ...},
        "versions": {"python": "3.12.10", "numpy": "2.4.4", ...}
      },
      ... the payload ...
    }

`git_dirty` matters more than the commit: a clean hash on a dirty tree is a false assurance, so
both are recorded and the flag is deliberately prominent.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]

#: Environment variables that change numerical results. Recorded whether or not they are set,
#: because "unset" is itself the fact that explains a number.
TRACKED_ENV = ("NCG_NORM", "NCG_CLIP", "NCG_MEMMAP", "NCG_NUM_WORKERS", "NCG_CACHE_BANDS",
               "NCG_HYBRID", "NCG_SSL_CKPT", "NCG_ORDINAL_STAGING", "NCG_LABEL_SMOOTH",
               "NCG_BLIND_TRAIN", "NCG_NO_GRAPH")


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:                                                     # noqa: BLE001
        return None


def _versions() -> Dict[str, str]:
    v = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "scipy", "sklearn", "torch", "mne", "mne_icalabel", "xgboost"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:                                                 # noqa: BLE001
            v[mod] = "absent"
    return v


def provenance(script: Optional[str] = None, extra: Optional[Dict[str, Any]] = None
               ) -> Dict[str, Any]:
    """Build the provenance block. Cheap enough to call on every write."""
    status = _git("status", "--porcelain")
    block: Dict[str, Any] = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "script": script or Path(sys.argv[0]).name,
        "argv": sys.argv[1:],
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "env": {k: os.environ.get(k) for k in TRACKED_ENV},
        "versions": _versions(),
    }
    if extra:
        block.update(extra)
    return block


def write_result(path, payload: Dict[str, Any], script: Optional[str] = None,
                 extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write `payload` as JSON with a provenance block attached. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if "provenance" in payload:
        raise ValueError("payload already carries a 'provenance' key; refusing to overwrite it")
    out = {"provenance": provenance(script, extra), **payload}
    path.write_text(json.dumps(out, indent=2, default=float))
    return path
