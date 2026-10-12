"""[B1] Depth models: frames/ -> depth_raw/ (float32 .npy, larger = closer)

Two models (choose with depth.model in the config):

  vda  = Video Depth Anything, Small (MAIN model, Apache 2.0)
         Looks at 32 frames at once and has a temporal head, so neighbouring
         frames get consistent depth.
  dav2 = Depth Anything V2, Small (BASELINE only, Apache 2.0)
         Looks at ONE frame at a time. Each frame is good on its own, but the
         scale/shift differs per frame, so the video flickers. Kept only to
         show how much better the video model is.

Both output relative "disparity": larger = closer, unknown scale and shift.

Long videos and 4K (design decision 7):
  * Frames are read in CHUNKS of depth.chunk frames (default 120) with
    depth.overlap frames shared between chunks. Each new chunk is scaled to
    match the previous one on the shared frames (least squares), so there is
    no jump. RAM use stays the same for 100 or 10,000 frames.
  * Frames taller than depth.max_height (default 1080) are shrunk before the
    model and the depth is upsampled back. The model only sees ~518 px
    anyway, so 4K input gives no extra detail, only extra memory. The
    fusion stage then re-sharpens the depth edges at full resolution.

    python run_pipeline.py --config configs/tos_08_3a.yaml --stages depth
    python run_pipeline.py --config configs/tos_08_3a.yaml --stages depth --set depth.model=dav2 depth.out_subdir=depth_raw_dav2
"""
import cv2
import numpy as np
import torch
from tqdm import tqdm

from common.io import frame_name, list_indices, read_rgb, write_depth
from common.third_party import add_repo, checkpoint, get_device
from common.timing import timed

ENCODER_CFG = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
}


def load_frames(paths, indices, max_height):
    """RGB frames, shrunk so height <= max_height (keeps aspect ratio)."""
    out = []
    for i in indices:
        f = read_rgb(paths.frames / frame_name(i, "png"))
        if max_height and f.shape[0] > max_height:
            s = max_height / f.shape[0]
            f = cv2.resize(f, (round(f.shape[1] * s), max_height), interpolation=cv2.INTER_AREA)
        out.append(f)
    return np.stack(out)


def to_full_res(d, H, W):
    if d.shape != (H, W):
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
    return d.astype(np.float32)


def load_vda(dc, device):
    add_repo("Video-Depth-Anything")
    from video_depth_anything.video_depth import VideoDepthAnything

    model = VideoDepthAnything(**ENCODER_CFG[dc["encoder"]])
    model.load_state_dict(torch.load(checkpoint(f"video_depth_anything_{dc['encoder']}.pth"), map_location="cpu"),
                          strict=True)
    return model.to(device).eval()


def run_vda(paths, indices, out_dir, dc, device, H, W):
    model = load_vda(dc, device)
    chunk, overlap = dc["chunk"], dc["overlap"]
    if overlap >= chunk:
        raise ValueError("depth.overlap must be smaller than depth.chunk")
    # The last `overlap` frames of each chunk are NOT written yet ("pending"): the next chunk
    # computes the same frames again, and we cross-fade old -> new over them. A hard switch
    # from one chunk to the next gave a visible depth pop (5x the normal frame-to-frame change).
    pending = None            # (ids, aligned depths) waiting for the next chunk
    start = 0                 # position of the first NEW frame of this chunk
    while start < len(indices):
        lo = max(0, start - overlap)
        ids = indices[lo:start + chunk]
        frames = load_frames(paths, ids, dc["max_height"])
        depths, _ = model.infer_video_depth(frames, target_fps=24, input_size=dc["input_size"],
                                            device=device.type, fp32=not dc["fp16"])
        n_shared = start - lo
        if pending is not None and n_shared > 0:
            old = pending[1][-n_shared:]
            # 1) fit s, b so this chunk matches the previous one on the shared frames
            a, b = depths[:n_shared].ravel()[::7], old.ravel()[::7]
            s, sh = np.linalg.lstsq(np.stack([a, np.ones_like(a)], 1), b, rcond=None)[0]
            depths = depths * s + sh
            # 2) cross-fade: weight of the new chunk goes 0 -> 1 across the shared frames
            w = ((np.arange(n_shared) + 1) / (n_shared + 1))[:, None, None]
            depths[:n_shared] = (1 - w) * old + w * depths[:n_shared]
        last = start + chunk >= len(indices)
        keep = 0 if last else min(overlap, len(ids))   # hold back the tail unless this is the end
        for i, d in zip(ids[:len(ids) - keep], depths[:len(ids) - keep]):
            write_depth(out_dir / frame_name(i, "npy"), to_full_res(d, H, W))
        pending = (ids, depths)
        start = lo + len(ids)
        torch.cuda.empty_cache()


def run_dav2(paths, indices, out_dir, dc, device, H, W):
    add_repo("Depth-Anything-V2")
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(**ENCODER_CFG[dc["encoder"]])
    model.load_state_dict(torch.load(checkpoint(f"depth_anything_v2_{dc['encoder']}.pth"), map_location="cpu"))
    model = model.to(device).eval()
    with torch.autocast(device_type=device.type, enabled=dc["fp16"] and device.type == "cuda"):
        for i in tqdm(indices, desc="DA-V2"):
            f = load_frames(paths, [i], dc["max_height"])[0]
            d = model.infer_image(np.ascontiguousarray(f[..., ::-1]), dc["input_size"])   # it expects BGR
            write_depth(out_dir / frame_name(i, "npy"), to_full_res(d, H, W))


def run(cfg, paths):
    dc = cfg["depth"]
    if dc["encoder"] != "vits":
        raise ValueError("only the Small (vits) models are Apache 2.0; Base/Large are CC-BY-NC (see CLAUDE.md)")
    out_dir = paths.root / dc["out_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = list_indices(paths.frames, "png")
    H, W = read_rgb(paths.frames / frame_name(indices[0], "png")).shape[:2]

    device = get_device()
    fn = {"vda": run_vda, "dav2": run_dav2}[dc["model"]]
    with timed(f"depth_{dc['model']}", paths.root, len(indices),
               {"input_size": dc["input_size"], "frame_size": f"{W}x{H}"}):
        fn(paths, indices, out_dir, dc, device, H, W)
    print(f"[depth] {dc['model']}: wrote {len(indices)} depth maps to {out_dir}")
