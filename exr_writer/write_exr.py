"""[D] EXR writer: frames/ (or source_exr/) + alpha/ + depth_final/ -> exr/0001.exr

Each EXR has 5 channels (design decisions 1, 4, 6):
    R, G, B  half float, LINEAR colour (VFX convention)
    A        half float, the object matte from MatAnyone (NOT multiplied into Z)
    Z        32-bit float, the FULL depth map of the whole frame

Depth direction (decision 4): inside the pipeline depth is normalised
disparity, larger = closer. HERE, and only here, it is converted:
    exr.z_mode = depth      -> Z = distance, larger = farther (Blender/Nuke style)
    exr.z_mode = disparity  -> Z = pipeline value, larger = closer
The depth is relative, not metric. For z_mode=depth we map the shot's
disparity range [0, 1] to distances [z_near, z_far] (default 1 .. 10
"scene units") with Z = 1 / disparity, because real disparity is 1/distance.

RGB is NOT premultiplied by default (exr.premultiply: false): RGB is the
full plate and A is the matte. In Nuke, add a Premult node to cut the
object out; in Blender, use "Straight" alpha.

Per-frame Z stats from fusion (stats.json) go into the EXR header too.

    python run_pipeline.py --config configs/dummy.yaml --stages exr
"""
import json

import numpy as np
import OpenEXR
from tqdm import tqdm

from common.io import frame_name, list_indices, read_alpha, read_depth, read_rgb
from common.timing import timed

COMPRESSION = {
    "none": OpenEXR.NO_COMPRESSION,
    "zip": OpenEXR.ZIP_COMPRESSION,
    "piz": OpenEXR.PIZ_COMPRESSION,
}


def disparity_to_z(disp: np.ndarray, ec: dict) -> np.ndarray:
    """The ONE place where pipeline depth (larger = closer) becomes the EXR Z channel."""
    disp = np.asarray(disp, np.float32)
    if ec["z_mode"] == "disparity":
        return disp
    if ec["z_mode"] != "depth":
        raise ValueError(f"exr.z_mode must be depth or disparity, not {ec['z_mode']!r}")
    near, far = float(ec["z_near"]), float(ec["z_far"])
    # disparity 0 -> 1/far, disparity 1 -> 1/near; clamp so far-away pixels stay finite
    inv = 1.0 / far + np.clip(disp, -0.2, 1.5) * (1.0 / near - 1.0 / far)
    inv = np.maximum(inv, 1.0 / (4.0 * far))
    return (1.0 / inv).astype(np.float32)


def srgb_to_linear(rgb8: np.ndarray) -> np.ndarray:
    x = rgb8.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def load_linear_rgb(paths, i, ec):
    """Prefer the untouched camera EXR (Tears of Steel), else convert the PNG."""
    src = paths.root / "source_exr" / frame_name(i, "exr")
    if ec.get("use_source_exr", True) and src.exists():
        from common.get_tears_of_steel import read_exr_rgb

        return read_exr_rgb(src)
    rgb8 = read_rgb(paths.frames / frame_name(i, "png"))
    return srgb_to_linear(rgb8) if ec["linearize_rgb"] else rgb8.astype(np.float32) / 255.0


def write_one(path, rgb, alpha, z, ec, extra_header=None):
    if ec.get("premultiply", False):
        rgb = rgb * alpha[..., None]
    header = {
        "compression": COMPRESSION[ec["compression"]],
        "type": OpenEXR.scanlineimage,
        "z_mode": ec["z_mode"],
        "depth_note": "relative depth, not metric; A is not multiplied into Z",
    }
    header.update(extra_header or {})
    channels = {
        "R": rgb[..., 0].astype(np.float16),
        "G": rgb[..., 1].astype(np.float16),
        "B": rgb[..., 2].astype(np.float16),
        "A": alpha.astype(np.float16),
        "Z": z.astype(np.float32),
    }
    with OpenEXR.File(header, channels) as f:
        f.write(str(path))


def run(cfg, paths):
    ec = cfg["exr"]
    indices = list_indices(paths.frames, "png")
    depth_dir = paths.depth_final if list_indices(paths.depth_final, "npy") else paths.depth
    if depth_dir is paths.depth:
        print("[exr] depth_final/ is empty, using depth/ (fusion stage skipped?)")
    stats = {}
    if paths.stats_json.exists():
        stats = json.loads(paths.stats_json.read_text(encoding="utf-8")).get("frames", {})

    with timed("exr", paths.root, len(indices)):
        for i in tqdm(indices, desc="EXR"):
            rgb = load_linear_rgb(paths, i, ec)
            alpha = read_alpha(paths.alpha / frame_name(i, "png"))
            z = disparity_to_z(read_depth(depth_dir / frame_name(i, "npy")), ec)
            if not (rgb.shape[:2] == alpha.shape == z.shape):
                raise ValueError(f"frame {i}: size mismatch rgb {rgb.shape} alpha {alpha.shape} z {z.shape}")
            extra = {}
            for k, v in stats.get(f"{i:04d}", {}).get("z", {}).items():
                if v is not None and k != "pixels":
                    extra[f"Z_{k}"] = float(v)        # e.g. Z_min, Z_max, Z_mean, Z_p02, Z_p98
            write_one(paths.exr / frame_name(i, "exr"), rgb, alpha, z, ec, extra)
    print(f"[exr] wrote {len(indices)} EXR files to {paths.exr}")
