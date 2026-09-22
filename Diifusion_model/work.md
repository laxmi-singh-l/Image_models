# Work Log — Diffusion (SD x4 Upscaler)

## Status
Done. `infer.py` runs `stabilityai/stable-diffusion-x4-upscaler` (fixed 4x).

## Files
- `infer.py` — standalone diffusion upscaling
- `theory.txt` — diffusion (SR3-style) super-resolution explainer

## What was done
- Wrapped `StableDiffusionUpscalePipeline` in a single `enhance()`
- CUDA: fp16 variant, attention slicing + model offload enabled to bound VRAM
- CPU: float32 pipeline
- Seeded generator for repeatable results; `guidance_scale=0` (pure upscale, no prompt conditioning)
- Matching `main_test.py` expectations: extra kwargs `steps`, `seed`

## Usage
```bash
python3 infer.py --input low_res.png --output out.png
python3 infer.py --input low_res.png --steps 50 --seed 42
```

## Notes / caveats
- Fixed to 4x (`SCALES = [4]`); passing any other scale aborts.
- First run downloads ~3.5 GB fp16 weights (CreativeML OpenRAIL-M).
- Slow — several seconds per image even on GPU; realistic micro-details, can hallucinate.