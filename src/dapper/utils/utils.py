# Generic dapper functions
import os
import shutil
from pathlib import Path
import subprocess

# Pathing for convenience
import dapper

_ROOT_DIR = Path(next(iter(dapper.__path__))).parent.parent
_DATA_DIR = _ROOT_DIR / "docs" / "data"


def _rm_and_mkdir(p: Path):
    if p.exists():
        for f in p.glob("*"):
            f.unlink()
    else:
        p.mkdir(parents=True, exist_ok=True)

def _rm(p: Path):
    if p.exists():
        for f in p.glob("*"):
            f.unlink()
        p.rmdir()


def make_directory(path, delete_all_contents=False):

    if os.path.isdir(path) is False:
        os.mkdir(path)
    elif delete_all_contents:
        remove_directory_contents(path)
    return


def remove_directory_contents(path, remove_directory=False):
    path = Path(path)

    # Bail out if the directory doesn't exist
    if not path.exists():
        return

    # Remove children
    for item in path.glob("*"):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    # Remove the directory itself if requested
    if remove_directory:
        path.rmdir()


def display_image_gh_notebook(image_file, alt="default"):
    """
    This creates an image to embed into Jupyter notebooks since live links
    are not displaying on Github, presumably due to the fact that the repo
    is private. Since it will remain private for the near future, I imagine this function
    will get plenty of use.

    Provide the image name as it appears in the notebooks/notebook_data/images directory.
    """
    import base64

    image_path = (
        _ROOT_DIR / "docs" / "notebooks" / "notebook_data" / "images" / image_file
    )
    # Read the image and convert it to Base64
    with open(image_path, "rb") as f:
        img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode("utf-8")

    html_img = f'<img src="data:image/jpeg;base64,{img_base64}" alt={alt} />'

    return html_img


def get_git_commit_hash():
    """
    Attempts to fetch the latest commit hash for the dapper repo. This is
    a bit hacky and not robust. It looks in the current working directory
    and works its way up directories until it finds a .git/ folder.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"
