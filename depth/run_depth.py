"""[B1] Depth models: frames/ -> depth_raw/ (float32 .npy, larger = closer)

Two models (choose with depth.model in the config):

  vda  = Video Depth Anything, Small (MAIN model, Apache 2.0)
         Looks at 32 frames at once and has a temporal head, so neighbouring
         frames get consistent depth. Long videos are processed in windows of
         32 frames with 10 overlapping, and each window is scale-aligned to
         the previous one, so there is no jump between windows.
  dav2 = Depth Anything V2, Small (BASELINE only, Apache 2.0)
         Looks at ONE frame at a time. Each frame is good on its own, but the
         scale/shift differs per frame, so the video flickers. Kept only to
         show how much better the video model is.

Both output relative "disparity": larger = closer, unknown scale and shift.
The models run at ~518 px (input_size) and the depth is upsampled back to
the frame size, which is why the fusion stage sharpens the edges later.

    python run_pipeline.py --config configs/tos_08_3a.yaml --stages depth
    # baseline into a separate folder:
    python run_pipeline.py --config configs/tos_08_3a_dav2.yaml --stages depth
"""
import numpy as np
import torch
from tqdm import tqdm

from common.io import frame_name, list_indices, read_rgb, write_depth
from common.third_party import add_repo, checkpoint, get_device
from common.timing import timed

ENCODER_CFG = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
}


def run_vda(frames, dc, device):
    add_repo("Video-Depth-Anything")
    from video_depth_anything.video_depth import VideoDepthAnything

    model = VideoDepthAnything(**ENCODER_CFG[dc["encoder"]])
    model.load_state_dict(torch.load(checkpoint(f"video_depth_anything_{dc['encoder']}.pth"), map_location="cpu"),
                          strict=True)
    model = model.to(device).eval()
    depths, _ = model.infer_video_depth(frames, target_fps=24, input_size=dc["input_size"],
                                        device=device.type, fp32=not dc["fp16"])
    return depths


def run_dav2(frames, dc, device):
    add_repo("Depth-Anything-V2")
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(**ENCODER_CFG[dc["encoder"]])
    model.load_state_dict(torch.load(checkpoint(f"depth_anything_v2_{dc['encoder']}.pth"), map_location="cpu"))
    model = model.to(device).eval()
    out = []
    with torch.autocast(device_type=device.type, enabled=dc["fp16"] and device.type == "cuda"):
        for f in tqdm(frames, desc="DA-V2"):
            out.append(model.infer_image(np.ascontiguousarray(f[..., ::-1]), dc["input_size"]))  # it expects BGR
    return np.stack(out)


def run(cfg, paths):
    dc = cfg["depth"]
    if dc["encoder"] != "vits":
        raise ValueError("only the Small (vits) models are Apache 2.0; Base/Large are CC-BY-NC (see CLAUDE.md)")
    out_dir = paths.root / dc["out_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = list_indices(paths.frames, "png")
    frames = np.stack([read_rgb(paths.frames / frame_name(i, "png")) for i in indices])   # N x H x W x 3, RGB

    device = get_device()
    fn = {"vda": run_vda, "dav2": run_dav2}[dc["model"]]
    with timed(f"depth_{dc['model']}", paths.root, len(indices), {"input_size": dc["input_size"]}):
        depths = fn(frames, dc, device)
    for i, d in zip(indices, depths):
        write_depth(out_dir / frame_name(i, "npy"), d.astype(np.float32))
    print(f"[depth] {dc['model']}: wrote {len(indices)} depth maps to {out_dir}")
