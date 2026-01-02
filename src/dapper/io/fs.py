from __future__ import annotations

import os
from pathlib import Path


def make_directory(path: str | Path, delete_all_contents: bool = False) -> Path:
    """Create a directory (optionally clearing its contents)."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    if delete_all_contents:
        remove_directory_contents(p, remove_directory=False)
    return p


def remove_directory_contents(path: str | Path, *, remove_directory: bool = False) -> None:
    """Remove all files/subdirs under `path`. Optionally remove the directory itself."""
    p = Path(path)
    if not p.exists():
        return

    for child in p.iterdir():
        if child.is_dir():
            # recursive
            for sub in child.rglob("*"):
                if sub.is_file() or sub.is_symlink():
                    try:
                        sub.unlink()
                    except FileNotFoundError:
                        pass
            # remove empty dirs bottom-up
            for sub in sorted([d for d in child.rglob("*") if d.is_dir()], key=lambda d: len(d.parts), reverse=True):
                try:
                    sub.rmdir()
                except OSError:
                    pass
            try:
                child.rmdir()
            except OSError:
                pass
        else:
            try:
                child.unlink()
            except FileNotFoundError:
                pass

    if remove_directory:
        try:
            p.rmdir()
        except OSError:
            pass
