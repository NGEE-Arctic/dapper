"""Deprecated module.

This project is being reorganized. Prefer:
- dapper.io.fs
- dapper.io.attrs
- dapper.io.provenance
"""

from __future__ import annotations

from dapper.io.fs import make_directory, remove_directory_contents
from dapper.io.attrs import apply_append_attrs
from dapper.io.provenance import get_git_commit_hash

__all__ = [
    "make_directory",
    "remove_directory_contents",
    "apply_append_attrs",
    "get_git_commit_hash",
]
