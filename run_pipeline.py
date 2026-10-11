"""One command: video -> EXR (R, G, B, A, Z).

    python run_pipeline.py --config configs/<shot>.yaml
    python run_pipeline.py --config configs/<shot>.yaml --stages depth align   # run only some stages
    python run_pipeline.py --config configs/<shot>.yaml --stages depth --set depth.model=dav2 depth.out_subdir=depth_raw_dav2

Each stage is a function `run(cfg, paths)` in its own module folder. Stages
are imported only when they run, so the CPU stages (align, fuse, exr) work
even on a machine without a GPU or PyTorch.
"""
import argparse
import importlib
import json
import time

import yaml

from common.config import load_config, set_seed
from common.io import ShotPaths

# stage name -> "module:function"
STAGES = {
    "extract": "common.extract_frames:extract",   # video -> frames/
    "segment": "segmentation.sam2_mask:run",      # frames/ -> mask/
    "matte": "matting.matanyone_alpha:run",       # frames/ + mask/ -> alpha/
    "depth": "depth.run_depth:run",               # frames/ -> depth_raw/
    "align": "alignment.align_depth:run",         # depth_raw/ -> depth/
    "fuse": "fusion.fuse:run",                    # alpha/ + depth/ + frames/ -> depth_final/, stats.json
    "exr": "exr_writer.write_exr:run",            # frames/ + alpha/ + depth_final/ -> exr/
}


def load_stage(name):
    module_name, func_name = STAGES[name].split(":")
    return getattr(importlib.import_module(module_name), func_name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="*", help="override the stage list in the config")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config values, e.g. --set depth.model=dav2 depth.out_subdir=depth_raw_dav2")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for item in args.set:
        key, value = item.split("=", 1)
        *parents, last = key.split(".")
        node = cfg
        for p in parents:
            node = node[p]
        node[last] = yaml.safe_load(value)   # "720" -> 720, "true" -> True, "abc" -> "abc"
    set_seed(cfg["seed"])
    paths = ShotPaths.from_config(cfg)
    paths.make_dirs()
    # keep a copy of the exact config next to the outputs (reproducibility)
    (paths.root / "config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    stages = args.stages or cfg["stages"]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; choose from {list(STAGES)}")

    t0 = time.perf_counter()
    for name in stages:
        print(f"\n===== {name} =====")
        fn = load_stage(name)
        if name == "extract":
            fn(cfg)
        else:
            fn(cfg, paths)
    print(f"\nDone in {time.perf_counter() - t0:.1f}s -> {paths.root}")


if __name__ == "__main__":
    main()
