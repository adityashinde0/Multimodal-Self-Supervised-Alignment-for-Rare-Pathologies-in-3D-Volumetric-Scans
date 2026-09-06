import os
import random
import subprocess
import sys
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets deterministic random seeds across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_git_commit():
    """
    Retrieves the current git commit hash if available.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except Exception:
        return "unversioned"


def get_reproducibility_metadata(seed=42):
    """
    Gathers environment and reproducibility metadata.
    """
    return {
        "git_commit": get_git_commit(),
        "seed": seed,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
