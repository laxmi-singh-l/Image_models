"""Diffusion-based super-resolution -- SD-X4 Upscaler (pre-trained, 4x).

Pipeline (see theory.txt):
  conditional iterative denoising -- the low-resolution image guides a U-Net
  that, step by step, turns pure gaussian noise into a crisp high-res image with
  realistic micro-details. This is the same family of method as SR3.

Weights (stabilityai/stable-diffusion-x4-upscaler, CreativeML OpenRAIL-M) are
auto-downloaded from Hugging Face Hub on first run (~3.5 GB fp16).

Usage:
    python3 infer.py --input low_res.png --output out.png
"""

import argparse
import logging
import os
import sys
import time
import warnings

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from utils import image_utils as iu  # noqa: E402

TITLE = "Diffusion (SD x4 Upscaler)"
SCALES = [4]  # this upscaler is fixed to 4x

_REPO = "stabilityai/stable-diffusion-x4-upscaler"


def _pick_device(device):
    import torch

    if device in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def enhance(input_path, output_path, scale=4, device="auto", steps=20, seed=0,
            prompt="", verbose=True):
    """Upscale `input_path` with a diffusion model and save to `output_path`."""
    import torch
    from diffusers import StableDiffusionUpscalePipeline

    if scale != 4:
        raise SystemExit("the diffusion upscaler is fixed to 4x (got scale=%s)" % scale)
    input_path = iu.resolve_input(input_path, _PROJECT, _HERE)
    dev = _pick_device(device)
    t0 = time.time()

    use_fp16 = dev.type == "cuda"
    dtype = torch.float16 if use_fp16 else torch.float32

    if verbose:
        print("[%s] loading weights from %s ..." % (TITLE, _REPO))
    kwargs = {"dtype": dtype}
    if use_fp16:
        kwargs["variant"] = "fp16"
    pipe = StableDiffusionUpscalePipeline.from_pretrained(_REPO, **kwargs)
    if dev.type == "cuda":
        pipe.enable_attention_slicing()
        # A full-frame VAE decode at 4x does not fit a 4 GB GPU, so tile it.
        try:
            pipe.enable_vae_tiling()
        except AttributeError:
            pipe.vae.enable_tiling()
        # Offload each submodule after use instead of holding everything on
        # the GPU (must be the last pipe configuration call).
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to(dev)
    else:
        pipe = pipe.to("cpu")

    lr = Image.open(input_path).convert("RGB")
    w, h = lr.size

    gen = torch.Generator(device=dev.type).manual_seed(seed)
    if verbose:
        print("[%s] denoising %d steps ..." % (TITLE, steps))
    _diffusers_log = logging.getLogger("diffusers")
    _diffusers_level = _diffusers_log.level
    _diffusers_log.setLevel(logging.ERROR)
    with torch.inference_mode():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`upcast_vae` is deprecated")
            warnings.filterwarnings(
                "ignore", message="modules in AutoencoderKL that should be kept in float32")
            result = pipe(prompt=prompt, image=lr, num_inference_steps=steps,
                          guidance_scale=0, generator=gen).images[0]
    _diffusers_log.setLevel(_diffusers_level)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    # tiled VAE decode can come back a few px short of the exact 4x size
    if result.size != (w * 4, h * 4):
        result = result.resize((w * 4, h * 4), Image.LANCZOS)
    result.save(output_path)
    if verbose:
        print("[%s] %dx%d -> %sx%s  (%.2fs)"
              % (TITLE, w, h, result.size[0], result.size[1], time.time() - t0))
        print("[%s] saved to %s" % (TITLE, output_path))
    return output_path


def main():
    p = argparse.ArgumentParser(description="Diffusion super-resolution (SD x4 Upscaler)")
    p.add_argument("--input", "-i", required=True, help="low-resolution input image")
    p.add_argument("--output", "-o", default=None, help="output image path")
    p.add_argument("--scale", type=int, default=4, choices=SCALES, help="upscale factor")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--steps", type=int, default=20, help="number of denoising steps")
    p.add_argument("--seed", type=int, default=0, help="manual seed for repeatability")
    p.add_argument("--prompt", default="", help="optional caption the model follows")
    a = p.parse_args()

    if a.output is None:
        root, ext = os.path.splitext(a.input)
        a.output = "%s_diffusion_x%d.png" % (root, a.scale)

    enhance(a.input, a.output, scale=a.scale, device=a.device,
            steps=a.steps, seed=a.seed, prompt=a.prompt)


if __name__ == "__main__":
    main()