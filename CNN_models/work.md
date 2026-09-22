# Work Log — CNN (EDSR)

## Status
Done. `infer.py` runs EDSR-base from Hugging Face at scales 2/3/4.

## Files
- `infer.py` — standalone EDSR inference (auto-downloads `eugenesiow/edsr-base`)
- `theory.txt` — CNN super-resolution explainer
- weights are fetched from the Hub on first run (not stored locally)

## What was done
- Wrapped `super_image.EdsrModel` with a single `enhance()` entry point
- CUDA auto-detection (`auto`/`cuda`/`cpu`) + fp16 half-precision on GPU
- Matching `main_test.py` expectations: `enhance(input_path, output_path, scale, device, verbose)`

## Usage
```bash
python3 infer.py --input low_res.png --scale 4 --output out.png
```

## Notes / caveats
- Scale must be in `SCALES = [2, 3, 4]`.
- Output is clean/smooth (L2-trained); alternative x2/x3 weights ship in the same checkpoint.