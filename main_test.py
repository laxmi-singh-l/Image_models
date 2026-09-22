#!/usr/bin/env python3
"""Run every model in the 4 folders on one low-res image and compare them.

Inspired by SwinIR's main_test_swinir.py: it upscales a query image with each
architecture, writes the individual results under results/<model>/, builds a
side-by-side comparison grid, and (optionally) reports PSNR/SSIM against a
ground-truth HR image.

Usage:
    python3 main_test.py --input path/to/low_res.png --scale 4
    python3 main_test.py --input lr.png --scale 4 --gt hr.png --device cuda
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils import image_utils as iu  # noqa: E402

# name -> model folder / which scales it supports natively
MODELS = [
    ("CNN (EDSR)",          "CNN_models",       [2, 4], {}),
    ("GANs (Real-ESRGAN)",  "GANs_model",       [2, 4], {"tile": 0}),
    ("ViTs (Swin2SR)",      "ViTs_model",       [2, 4], {}),
    ("Diffusion (SD-x4)",   "Diifusion_model",  [4],    {"steps": 20, "seed": 0}),
]


def _load_enhance(subdir):
    path = os.path.join(_HERE, subdir, "infer.py")
    model_dir = os.path.dirname(path)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    spec = importlib.util.spec_from_file_location(subdir + "_infer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.enhance


def _opts():
    p = argparse.ArgumentParser(description="Compare all 4 super-resolution models (SwinIR-style)")
    p.add_argument("--input", "-i", required=True, help="low-resolution input image")
    p.add_argument("--scale", type=int, default=4, choices=[2, 4], help="upscale factor")
    p.add_argument("--gt", default=None, help="ground-truth HR image (same scene, LR*scale size)")
    p.add_argument("--outdir", default="results", help="output directory")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--steps", type=int, default=20, help="diffusion denoising steps")
    p.add_argument("--seed", type=int, default=0, help="diffusion seed")
    p.add_argument("--cell", type=int, default=320, help="cell width of the comparison grid")
    p.add_argument("--zoom", type=int, default=4, help="zoom factor for the center-crop inset")
    return p.parse_args()


def main():
    a = _opts()
    a.input = iu.resolve_input(a.input, _HERE)
    os.makedirs(a.outdir, exist_ok=True)

    lr = iu.read_image_rgb(a.input)
    lr_padded, pad = iu.make_divisible(lr, a.scale)
    lr_h, lr_w = lr.shape[:2]
    out_h, out_w = lr_h * a.scale, lr_w * a.scale

    stem = os.path.splitext(os.path.basename(a.input))[0]
    print("=" * 70)
    print("Super-resolution comparison @ scale x%d -- %s (%dx%d -> %dx%d)"
          % (a.scale, a.input, lr_w, lr_h, out_w, out_h))
    print("=" * 70)

    # ---- baseline: bicubic ----
    bic = iu.bicubic_upscale(lr_padded, a.scale)
    bic = iu.crop_back(bic, pad, a.scale)
    bic_out = os.path.join(a.outdir, "bicubic", "%s_bicubic_x%d.png" % (stem, a.scale))
    iu.write_image(bic_out, bic)

    # ---- every model ----
    entries = []
    results = []
    for title, subdir, scales, extra in MODELS:
        if a.scale not in scales:
            print("  * %-22s skipped (native scales %s)" % (title, scales))
            continue
        mod_out = os.path.join(a.outdir, subdir.lower().replace("_model", ""),
                               "%s_%s_x%d.png" % (stem, subdir.split("_")[0], a.scale))
        enh = _load_enhance(subdir)
        t = time.time()
        print("  [%s] running ..." % title)
        enh(input_path=a.input, output_path=mod_out, scale=a.scale,
            device=a.device, verbose=False, **extra)
        elapsed = time.time() - t
        out_img = Image.open(mod_out).convert("RGB")
        # guard: every output should end at LR*scale (in case of rounding)
        if out_img.size != (out_w, out_h):
            out_img = out_img.resize((out_w, out_h), Image.LANCZOS)
            out_img.save(mod_out)
        entries.append((title, out_img))
        results.append(dict(name=title, path=mod_out, elapsed=elapsed))
        print("       -> %-22s %dx%d  (%.2fs)  %s" % (title, out_w, out_h, elapsed, mod_out))

    # ---- comparison grid ----
    grid_path = os.path.join(a.outdir, "comparison_x%d.png" % a.scale)
    iu.build_comparison(grid_path, "Super-Resolution Comparison  x%d" % a.scale,
                        [("Bicubic\n(baseline)", Image.fromarray(bic))] + entries,
                        cell_width=a.cell, zoom_factor=a.zoom)
    print("comparison grid saved to %s" % grid_path)

    # ---- metrics (optional GT) ----
    if a.gt:
        import cv2

        gt = iu.read_image_rgb(a.gt)
        if gt.shape[:2] != (out_h, out_w):
            print("!! ground truth must be exactly %dx%d (got %s); resizing to compare."
                  % (out_w, out_h, gt.shape[::-1]))
            gt = cv2.resize(gt, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        metric_rows = [("Model", "PSNR(dB)", "SSIM", "Time")]
        # baseline
        psnr, ssim = iu.calc_psnr_ssim(gt, bic)
        metric_rows.append(("Bicubic (baseline)", psnr, ssim, "-"))
        for r in results:
            img = iu.read_image_rgb(r["path"])
            psnr, ssim = iu.calc_psnr_ssim(gt, img)
            metric_rows.append((r["name"], psnr, ssim, "%.2fs" % r["elapsed"]))
        print()
        table_txt = []
        for name, psnr, ssim, t in metric_rows:
            if isinstance(psnr, str):
                line = "%-22s %-10s %-8s %s" % (name, psnr, ssim, t)
            else:
                line = "%-22s %-10.4f %-8.4f %s" % (name, psnr, ssim, t)
            print(line)
            table_txt.append(line)
        with open(os.path.join(a.outdir, "metrics_x%d.txt" % a.scale), "w") as f:
            f.write("Ground truth: %s\n" % a.gt)
            f.write("Input: %s (LR)\n\n" % a.input)
            f.write("\n".join(table_txt) + "\n")

    print("done.")


if __name__ == "__main__":
    main()