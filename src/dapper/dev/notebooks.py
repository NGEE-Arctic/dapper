from __future__ import annotations

from pathlib import Path


def display_image_gh_notebook(image_file: str | Path, alt: str = "default") -> str:
    """Return a GitHub-friendly markdown string for embedding a local image in notebooks."""
    image_file = Path(image_file)
    # Use raw relative path in the markdown; caller controls location.
    return f"![{alt}]({image_file.as_posix()})"
