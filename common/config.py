"""Load a per-shot YAML config on top of configs/defaults.yaml.

The shot config only needs the values that differ from the defaults
(shot name, video path, SAM 2 prompts, ...).
"""
import copy
import random
from pathlib import Path

import numpy as np
import yaml

DEFAULTS = Path(__file__).resolve().parent.parent / "configs" / "defaults.yaml"


def _merge(base: dict, override: dict) -> dict:
    """Recursive dict merge: values in `override` win."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path) -> dict:
    with open(DEFAULTS, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(path, encoding="utf-8") as f:
        cfg = _merge(cfg, yaml.safe_load(f))
    if not cfg.get("shot"):
        raise ValueError(f"{path}: 'shot' is required")
    return cfg


def set_seed(seed: int):
    """Fix all random seeds so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except (ImportError, OSError):
        pass  # torch not installed or blocked; non-GPU stages still work
