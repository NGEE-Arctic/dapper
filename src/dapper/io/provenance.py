from __future__ import annotations

import subprocess


def get_git_commit_hash() -> str:
    """Return the current git commit hash if available, otherwise 'unknown'."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"
