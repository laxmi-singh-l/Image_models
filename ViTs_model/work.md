# Work Log — ViTs (Swin2SR)

## Status
Done. `infer.py` runs Swin2SR (Hugging Face `transformers`) at scales 2/4.

## Files
- `infer.py` — standalone Swin2SR inference
- `theory.txt` — ViT super-resolution explainer (shifted-window attention)

## What was done
- Wrapped `Swin2SRForImageSuperResolution` + `AutoImageProcessor` in a single `enhance()`
- Checkpoint map: x2 → `caidas/swin2SR-lightweight-x2-64`, x4 → `caidas/swin2SR-classical-sr-x4-64`
- CUDA auto-detection + fp16 half-precision on GPU
- Reverts the HuggingFace BGR→RGB channel-order convention before saving
- Pre-pads input to a multiple of `scale` (via `utils.image_utils`) so H/W stay exact

## Usage
```bash
python3 infer.py --input low_res.png --scale 4 --output out.png
```

## Notes / caveats
- Scale must be in `SCALES = [2, 4]`.
- Mathematically precise edges; good for text, architecture, structured patterns.