"""[A1] SAM 2 video segmentation: frames/ + prompts -> mask/ (binary).

How it works:
  1. You click a few points (or draw a box) on one frame; they live in the
     shot config under segmentation.prompts.
  2. SAM 2 turns the prompts into a mask on that frame, keeps a "memory" of
     what the object looks like, and propagates the mask through the video.
  3. If the object gets lost, add another prompt on a later frame: SAM 2
     uses every prompted frame as an extra memory.

The output is BINARY (0/255). It is only the starting point for MatAnyone,
not a final alpha (hair, motion blur and soft edges need matting).

    python run_pipeline.py --config configs/tos_08_3a.yaml --stages segment
"""
import shutil

import cv2
import numpy as np
import torch
from tqdm import tqdm

from common.io import frame_name, list_indices, write_mask
from common.third_party import add_repo, checkpoint, get_device
from common.timing import timed

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"   # matches facebook/sam2.1-hiera-small
SAM2_CKPT = "sam2.1_hiera_small.pt"


def _make_jpeg_folder(paths, indices):
    """SAM 2 only reads a folder of JPEGs named 00000.jpg, 00001.jpg, ... (0-based).
    Our frame 0001.png becomes 00000.jpg. Quality 95 keeps edges sharp."""
    jpg_dir = paths.root / "_cache" / "sam2_jpg"
    if jpg_dir.exists() and len(list(jpg_dir.glob("*.jpg"))) == len(indices):
        return jpg_dir
    shutil.rmtree(jpg_dir, ignore_errors=True)
    jpg_dir.mkdir(parents=True)
    for k, i in enumerate(indices):
        img = cv2.imread(str(paths.frames / frame_name(i, "png")))
        cv2.imwrite(str(jpg_dir / f"{k:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return jpg_dir


def run(cfg, paths):
    prompts = cfg["segmentation"]["prompts"]
    if not prompts:
        raise ValueError("segmentation.prompts is empty: add at least one point or box in the shot config")
    indices = list_indices(paths.frames, "png")
    jpg_dir = _make_jpeg_folder(paths, indices)

    add_repo("sam2")
    from sam2.build_sam import build_sam2_video_predictor

    device = get_device()
    predictor = build_sam2_video_predictor(SAM2_CONFIG, str(checkpoint(SAM2_CKPT)), device=device)

    with timed("segment", paths.root, len(indices)), torch.inference_mode(), \
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        # offload_video_to_cpu keeps the 100+ decoded frames in RAM, not in the 8 GB of VRAM
        state = predictor.init_state(video_path=str(jpg_dir), offload_video_to_cpu=True)

        for p in prompts:
            k = indices.index(p["frame"])          # our 1-based frame number -> SAM 2 0-based index
            pts = np.array(p["points"], np.float32) if p.get("points") else None
            lbl = np.array(p["labels"], np.int32) if p.get("points") else None
            box = np.array(p["box"], np.float32) if p.get("box") else None
            predictor.add_new_points_or_box(state, frame_idx=k, obj_id=1, points=pts, labels=lbl, box=box)

        masks = {}
        first = min(indices.index(p["frame"]) for p in prompts)
        # forward from the first prompted frame, then backward if it was not frame 1
        for reverse in ([False, True] if first > 0 else [False]):
            for k, obj_ids, logits in tqdm(predictor.propagate_in_video(state, start_frame_idx=first, reverse=reverse),
                                           desc="SAM 2", total=len(indices)):
                masks[k] = (logits[0, 0] > 0).cpu().numpy()

    empty = []
    for k, i in enumerate(indices):
        m = masks.get(k, np.zeros((state["video_height"], state["video_width"]), bool))
        if m.sum() == 0:
            empty.append(i)
        write_mask(paths.mask / frame_name(i, "png"), m)
    if empty:
        print(f"[segment] WARNING: object lost on {len(empty)} frames, e.g. {empty[:5]}. "
              "Add a prompt on one of those frames in the config.")
    print(f"[segment] wrote {len(indices)} masks to {paths.mask}")
