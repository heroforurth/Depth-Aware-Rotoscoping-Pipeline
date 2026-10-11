"""Make the cloned model repos in third_party/ importable, and find checkpoints.

We do NOT `pip install` the repos: Video Depth Anything pins torch 2.1 and
numpy 1.24, which would downgrade our CUDA 13 torch and break the RTX 50 GPU.
Instead each repo folder is put on sys.path, and only the small extra
packages they need are installed (see requirements.txt).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = ROOT / "third_party"
CHECKPOINTS = ROOT / "checkpoints"

REPOS = {
    "sam2": "https://github.com/facebookresearch/sam2.git",
    "MatAnyone": "https://github.com/pq-yang/MatAnyone.git",
    "Video-Depth-Anything": "https://github.com/DepthAnything/Video-Depth-Anything.git",
    "Depth-Anything-V2": "https://github.com/DepthAnything/Depth-Anything-V2.git",
}


def add_repo(name: str) -> Path:
    path = THIRD_PARTY / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run: powershell scripts/setup_models.ps1  (or git clone {REPOS[name]} {path})"
        )
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def checkpoint(filename: str) -> Path:
    path = CHECKPOINTS / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: powershell scripts/setup_models.ps1")
    return path


def get_device():
    import torch

    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU found, running on CPU (very slow)")
        return torch.device("cpu")
    return torch.device("cuda")
