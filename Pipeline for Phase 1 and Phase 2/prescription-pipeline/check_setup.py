"""
check_setup.py — verifies all files are in the right place BEFORE running the pipeline.
Run this first: python check_setup.py
It checks every path the pipeline needs and tells you exactly what's missing.
"""
from pathlib import Path
import sys

print("="*60)
print("PIPELINE SETUP CHECK")
print("="*60)

checks = {
    "segmentation_pipeline.py":            "segmentation_pipeline.py",
    "full_pipeline_v2.py":                 "full_pipeline_v2.py",
    "CRAFT weights":                       "models/craft_mlt_25k.pth",
    "CRAFT repo folder":                   "libs/CRAFT-pytorch",
    "TrOCR config":                        "checkpoints/trocr_augmented/best/config.json",
    "TrOCR weights":                       "checkpoints/trocr_augmented/best/model.safetensors",
    "TrOCR gen config":                    "checkpoints/trocr_augmented/best/generation_config.json",
    "Formulary CSV":                       "data/formulary/drug_names.csv",
    "Train CSV":                           "data/pharmacy_lk/splits/train.csv",
    "Test images folder":                  "data/test_images",
}

all_ok = True
for label, path in checks.items():
    p = Path(path)
    ok = p.exists()
    if not ok: all_ok = False
    print(f"  {'OK ' if ok else 'MISSING':9s} {label:24s} -> {path}")

# check test images present
ti = Path("data/test_images")
if ti.exists():
    imgs = list(ti.glob("*.jpg")) + list(ti.glob("*.png")) + list(ti.glob("*.jpeg"))
    print(f"\n  test images found: {len(imgs)}")
    for im in imgs[:5]:
        print(f"     {im.name}")

# check key dependencies
print("\n-- dependencies --")
for mod in ["torch", "torchvision", "cv2", "transformers", "PIL", "pandas", "skimage", "scipy"]:
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "?")
        print(f"  OK  {mod:14s} {v}")
    except ImportError:
        print(f"  MISSING  {mod}")
        all_ok = False

# transformers version check (critical)
try:
    import transformers
    if not transformers.__version__.startswith("4.40"):
        print(f"\n  !! WARNING: transformers is {transformers.__version__}, "
              f"but TrOCR needs 4.40.2. Run: pip install transformers==4.40.2")
except Exception:
    pass

print("\n" + "="*60)
print("ALL GOOD — ready to run full_pipeline_v2.py" if all_ok
      else "FIX THE MISSING ITEMS ABOVE before running the pipeline")
print("="*60)
