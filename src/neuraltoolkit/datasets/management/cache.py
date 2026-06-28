from pathlib import Path
import shutil
from warnings import warn

from numpy import stack
from neuraltoolkit.datasets.management.paths import DATASETS_PATH


def clear_cache():
    """deletes .ntk/datasets/*"""
    if DATASETS_PATH.exists():
        shutil.rmtree(DATASETS_PATH)

def cache_dir():
    """prints the location of the datasets directory"""
    if DATASETS_PATH.exists():
        print(f"NTK cache directory - {DATASETS_PATH}")
    else:
        warn(
            ("The cache directory doesn't exist as there are no datasets "
               "When a dataset is downloaded the directory will default to "
               "~/.ntk/datasets"), 
               RuntimeWarning,
               stacklevel=2)

def list():
    """lists the files and folders within .ntk/datasets"""

    if DATASETS_PATH.exists():
        print(f"From - {DATASETS_PATH}")
        for entry in DATASETS_PATH.iterdir():
            print(f"\t{entry.name}")
    else:
        warn(
            ".ntk/datasets does not exist and is therefore empty. Download a dataset to initialize this directory", 
            RuntimeWarning,
            stacklevel=2
            )

def delete(file_name:str):
    """Delete a file or folder from .ntk/datasets"""
    file = Path(file_name)

    for entry in DATASETS_PATH.iterdir():
        if entry.stem == file.stem:
            file_path = DATASETS_PATH / entry
            file_path.unlink(missing_ok=True)