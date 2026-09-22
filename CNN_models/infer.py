"""EDSR -- Convolutional Neural Network super-resolution (pre-trained).

Pipeline (see theory.txt):
  1. a single-pass CNN directly maps LR pixels -> HR pixels (residual learning), and
  2. the final upsampling is learned at the tail of the network.

Usage:
    python3 infer.py --input low_res.png --scale 4 --output out.png

First run auto-downloads the pre-trained weights from Hugging Face Hub.
"""

import argparse
import os
import sys
import time

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from utils import image_utils as iu  # noqa: E402

TITLE = "CNN (EDSR)"
SCALES = [2, 3, 4]

# this checkpoint ships a weight file per scale: pytorch_model_{scale}x.pt
_CHECKPOINT = "eugenesiow/edsr-base"


def _pick_device(device):
    import torch

    if device in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def enhance(input_path, output_path, scale=4, device="auto", verbose=True):
    """Upscale `input_path` with EDSR and save the result to `output_path`."""
    import torch
    from super_image import EdsrModel, ImageLoader

    if scale not in SCALES:
        raise SystemExit("EDSR supports scale in %s (got %s)" % (SCALES, scale))
    input_path = iu.resolve_input(input_path, _PROJECT, _HERE)
    dev = _pick_device(device)
    t0 = time.time()

    if verbose:
        print("[%s] loading %dx weights from %s ..." % (TITLE, scale, _CHECKPOINT))
    # The scale kwarg is honoured by super-image and selects the right weight
    # file (pytorch_model_{scale}x.pt) from the checkpoint's Hub repo.
    model = EdsrModel.from_pretrained(_CHECKPOINT, scale=scale)
    model = model.to(dev).eval()
    if dev.type == "cuda":
        model = model.half()

    lr = ImageLoader.load_image(Image.open(input_path).convert("RGB")).to(dev)
    if dev.type == "cuda":
        lr = lr.half()
    with torch.no_grad():
        pred = model(lr)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    # ImageLoader.save_image chokes on fp16 tensors (cv2 has no CV_16F path),
    # so normalise to RGB uint8 ourselves and save with PIL.
    from PIL import Image as PILImage
    PILImage.fromarray(iu.from_float_tensor(pred.float())).save(output_path)
    if verbose:
        print("[%s] %dx%d -> %dx%d  (%.2fs)"
              % (TITLE, lr.shape[-1], lr.shape[-2],
                 pred.shape[-1], pred.shape[-2], time.time() - t0))
        print("[%s] saved to %s" % (TITLE, output_path))
    return output_path


def main():
    p = argparse.ArgumentParser(description="EDSR super-resolution (CNN)")
    p.add_argument("--input", "-i", required=True, help="low-resolution input image")
    p.add_argument("--output", "-o", default=None, help="output image path")
    p.add_argument("--scale", type=int, default=4, choices=SCALES, help="upscale factor")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    a = p.parse_args()

    if a.output is None:
        root, ext = os.path.splitext(a.input)
        a.output = "%s_EDSR_x%d.png" % (root, a.scale)

    enhance(a.input, a.output, scale=a.scale, device=a.device)


if __name__ == "__main__":
    main()