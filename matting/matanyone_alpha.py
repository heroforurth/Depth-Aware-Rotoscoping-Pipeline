"""[A2] MatAnyone video matting: frames/ + first-frame mask/ -> alpha/ (soft, 0-1).

Why: a SAM 2 mask is binary (0 or 1). Real edges are not: hair strands,
motion blur and out-of-focus edges are partly transparent. Compositors
need that soft alpha, otherwise the cut-out looks like a paper sticker.

How it works:
  1. Take ONE binary mask (frame matting.mask_frame, default 1) from SAM 2.
  2. Shrink it a bit (erode) and grow it a bit (dilate): the band in between
     becomes "unknown", where MatAnyone decides the real soft edge.
  3. "Warm-up": run the first frame several times so MatAnyone settles on a
     good first alpha before it starts moving through the video.
  4. For every next frame, MatAnyone uses its memory of earlier frames to
     predict the alpha, so the matte stays stable over time.

Licence: MatAnyone is NTU S-Lab License 1.0 (non-commercial use).

    python run_pipeline.py --config configs/tos_08_3a.yaml --stages matte
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common.io import frame_name, list_indices, read_mask, read_rgb, write_alpha
from common.third_party import add_repo, checkpoint, get_device
from common.timing import timed


def _prepare_mask(mask: np.ndarray, erode: int, dilate: int) -> np.ndarray:
    """Same idea as MatAnyone's own demo, but with fixed (not random) kernel sizes,
    so runs are reproducible. Small holes from SAM 2 get closed by the dilate."""
    m = mask.astype(np.float32) * 255
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        m = cv2.dilate((m > 0).astype(np.float32), k) * 255
    if erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode))
        m = cv2.erode((m == 255).astype(np.float32), k) * 255
    return m


def run(cfg, paths):
    mc = cfg["matting"]
    indices = list_indices(paths.frames, "png")
    start = mc["mask_frame"]
    if start != indices[0]:
        # MatAnyone only runs forward; frames before the mask frame would get no alpha
        raise ValueError("matting.mask_frame must be the first frame for now")

    device = get_device()
    model = load_matanyone(device)
    frames = (read_rgb(paths.frames / frame_name(i, "png")) for i in indices)
    first_mask = read_mask(paths.mask / frame_name(start, "png"))
    with timed("matte", paths.root, len(indices), {"max_size": mc.get("max_size", -1)}):
        alphas = matte_video(model, frames, first_mask, mc["erode"], mc["dilate"], mc["warmup"],
                             mc.get("max_size", -1), device)
        for i, alpha in zip(tqdm(indices, desc="MatAnyone"), alphas):
            write_alpha(paths.alpha / frame_name(i, "png"), alpha)
    print(f"[matte] wrote {len(indices)} alpha mattes to {paths.alpha}")


def load_matanyone(device):
    """Load the MatAnyone network once; reuse it for many clips (evaluation does this)."""
    add_repo("MatAnyone")
    from hydra.core.global_hydra import GlobalHydra
    from matanyone.utils.get_default_model import get_matanyone_model

    GlobalHydra.instance().clear()   # SAM 2 may have initialised Hydra in this same process
    return get_matanyone_model(str(checkpoint("matanyone.pth")), device)


def matte_video(model, frames, first_mask, erode, dilate, warmup, max_size, device):
    """Yield one float32 alpha (H x W, 0-1) per frame.
    frames: iterable of RGB uint8 frames (first one = the mask's frame). first_mask: bool H x W."""
    from matanyone.inference.inference_core import InferenceCore

    processor = InferenceCore(model, cfg=model.cfg)          # fresh memory for every clip
    frames = iter(frames)
    first = next(frames)
    H, W = first.shape[:2]
    # optional downscale for memory: short side -> max_size, alpha is upsampled back
    scale = max_size / min(H, W) if 0 < max_size < min(H, W) else 1.0
    h, w = int(round(H * scale)), int(round(W * scale))

    mask = torch.from_numpy(_prepare_mask(first_mask, erode, dilate)).float().to(device)
    if scale != 1.0:
        mask = F.interpolate(mask[None, None], size=(h, w), mode="nearest")[0, 0]

    def to_tensor(rgb):
        img = torch.from_numpy(rgb).permute(2, 0, 1).float()
        if scale != 1.0:
            img = F.interpolate(img[None], size=(h, w), mode="area")[0]
        return (img / 255.0).to(device)

    def to_alpha(prob):
        alpha = processor.output_prob_to_mask(prob).float()
        if scale != 1.0:
            alpha = F.interpolate(alpha[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        return alpha.clamp(0, 1).cpu().numpy()

    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        img = to_tensor(first)
        processor.step(img, mask, objects=[1])                 # encode the given mask
        for _ in range(warmup + 1):                            # warm-up on frame 1
            prob = processor.step(img, first_frame_pred=True)
        yield to_alpha(prob)
        for rgb in frames:
            yield to_alpha(processor.step(to_tensor(rgb)))
