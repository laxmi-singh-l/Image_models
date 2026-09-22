"""Swin2SR -- Vision Transformer super-resolution (pre-trained).

Pipeline (see theory.txt):
  the LR image is split into patches (tokens); a shifted-window self-attention
  backbone models long-range pixel relationships, and a sub-pixel convolution
  projects the enriched tokens back onto an upscaled pixel grid.

Weights (caidas/Swin2SR, MIT licence on HF Hub) are auto-downloaded on first run.

Usage:
    python3 infer.py --input low_res.png --scale 4 --output out.png
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from utils import image_utils as iu  # noqa: E402

TITLE = "ViTs (Swin2SR)"
SCALES = [2, 4]

_CHECKPOINTS = {
    2: "caidas/swin2SR-lightweight-x2-64",
    4: "caidas/swin2SR-classical-sr-x4-64",
}


def _pick_device(device):
    import torch

    if device in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def enhance(input_path, output_path, scale=4, device="auto", verbose=True):
    """Upscale `input_path` with Swin2SR and save the result to `output_path`."""
    import torch
    from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

    if scale not in SCALES:
        raise SystemExit("Swin2SR supports scale in %s (got %s)" % (SCALES, scale))
    input_path = iu.resolve_input(input_path, _PROJECT, _HERE)
    dev = _pick_device(device)
    t0 = time.time()

    repo = _CHECKPOINTS[scale]
    if verbose:
        print("[%s] loading weights from %s ..." % (TITLE, repo))
    processor = AutoImageProcessor.from_pretrained(repo)
    model = Swin2SRForImageSuperResolution.from_pretrained(repo)
    model = model.to(dev).eval()
    if dev.type == "cuda":
        model = model.half()

    arr = np.asarray(Image.open(input_path).convert("RGB"))
    h, w = arr.shape[:2]
    arr, pad = iu.make_divisible(arr, scale)
    img = Image.fromarray(arr)

    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(dev)
    if dev.type == "cuda":
        pixel_values = pixel_values.half()
    with torch.no_grad():
        rec = model(pixel_values=pixel_values).reconstruction

    # reconstruction is returned in BGR channel order (HuggingFace convention)
    rec = rec.squeeze(0).float().clamp_(0, 1).cpu().numpy()
    sr = np.transpose(rec[[2, 1, 0], :, :], (1, 2, 0))
    sr = np.round(sr * 255.0).astype(np.uint8)
    # the model pads internally to its window size; trim back to scale*input
    target_h, target_w = arr.shape[0] * scale, arr.shape[1] * scale
    if sr.shape[0] < target_h or sr.shape[1] < target_w:
        sr = np.pad(sr, ((0, max(0, target_h - sr.shape[0])),
                         (0, max(0, target_w - sr.shape[1])), (0, 0)),
                    mode="edge")
    sr = sr[:target_h, :target_w]
    sr = iu.crop_back(sr, pad, scale)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    Image.fromarray(sr).save(output_path)
    if verbose:
        print("[%s] %dx%d -> %dx%d  (%.2fs)"
              % (TITLE, w, h, sr.shape[1], sr.shape[0], time.time() - t0))
        print("[%s] saved to %s" % (TITLE, output_path))
    return output_path


def main():
    p = argparse.ArgumentParser(description="Swin2SR super-resolution (Vision Transformer)")
    p.add_argument("--input", "-i", required=True, help="low-resolution input image")
    p.add_argument("--output", "-o", default=None, help="output image path")
    p.add_argument("--scale", type=int, default=4, choices=SCALES, help="upscale factor")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    a = p.parse_args()

    if a.output is None:
        root, ext = os.path.splitext(a.input)
        a.output = "%s_Swin2SR_x%d.png" % (root, a.scale)

    enhance(a.input, a.output, scale=a.scale, device=a.device)


if __name__ == "__main__":
    main()