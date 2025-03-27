# Generic dapper functions JPS
import os
import shutil
import requests
from pathlib import Path
from datetime import datetime

# Pathing for convenience
import dapper
_ROOT_DIR = Path(next(iter(dapper.__path__))).parent.parent
_DATA_DIR = _ROOT_DIR / "data"


def make_directory(path, delete_all_contents=False):

    if os.path.isdir(path) is False:
        os.mkdir(path)
    elif delete_all_contents:
        remove_directory_contents(path)
    return


def remove_directory_contents(path, remove_directory=False):
    if any(path.glob("*")):  # Check if directory contains any files
        for item in path.glob("*"):
            if item.is_file():
                item.unlink()  # Delete file
            elif item.is_dir():
                shutil.rmtree(item)  # Delete folder and its contents 

    if remove_directory:
         path.rmdir()
         

def display_image_gh_notebook(image_file, alt='default'):
    """
    This creates an image to embed into Jupyter notebooks since live links
    are not displaying on Github, presumably due to the fact that the repo
    is private. Since it will remain private for the near future, I imagine this function
    will get plenty of use.
    
    Provide the image name as it appears in the notebooks/notebook_data/images directory.
    """
    import base64
    from dapper.utils import _ROOT_DIR

    image_path = _ROOT_DIR / 'docs' / 'notebooks' / 'notebook_data' / 'images' / image_file
    # Read the image and convert it to Base64
    with open(image_path, 'rb') as f:
        img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')

    html_img = f'<img src="data:image/jpeg;base64,{img_base64}" alt={alt} />'

    return html_img


# def download_pangeo_cmip_catalog():
#     """
#     Downloads the Pangeo CMIP6 catalog. Run if you want to make sure you're working
#     with the latest version.
#     """
#     url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"

#     # Format today's date as YYYYMMDD
#     today = datetime.now().strftime("%Y%m%d")
#     path_out = _DATA_DIR / f"pangeo-cmip6-{today}.json"

#     # Download the file
#     response = requests.get(url)
#     response.raise_for_status()

#     # Save the file
#     with open(path_out, "wb") as f:
#         f.write(response.content)

#     print(f'Pangeo file saved at {path_out}.')