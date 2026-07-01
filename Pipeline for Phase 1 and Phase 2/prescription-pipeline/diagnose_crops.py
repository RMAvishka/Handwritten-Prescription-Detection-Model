"""
diagnose_crops.py — saves each crop with its RAW HTR output IN THE FILENAME,
so you can open one folder and instantly see what each crop contained vs what HTR read.

Output: output/diagnostic/  containing files like:
    line00_RAW-analron_SNAP-none_DIST-0.43.jpg
    line11_RAW-empa_SNAP-empa_DIST-0.0.jpg
Open that folder, look at each image, and check: does the crop's actual content
match the RAW output? That tells us if it's a segmentation or a TrOCR-reject problem.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import cv2
from pathlib import Path
from full_pipeline_v2 import PrescriptionPipeline

IMAGE = "data/test_images/176805_1.Jpg"   # <-- set your test image
OUT = Path("output/diagnostic")
OUT.mkdir(parents=True, exist_ok=True)

pipe = PrescriptionPipeline(
    craft_model_path="models/craft_mlt_25k.pth",
    craft_repo_dir="libs/CRAFT-pytorch",
    trocr_ckpt="checkpoints/trocr_augmented/best",
    formulary_csv="data/formulary/drug_names.csv",
    train_csv="data/pharmacy_lk/splits/train.csv",
)

res = pipe.run(IMAGE)
print(f"\nSaving {len(res['all_regions'])} annotated crops to {OUT}/\n")
for r in res["all_regions"]:
    raw = (r["raw_recognition"] or "empty")[:20]
    snap = (r["medicine_name"] or "none")[:20]
    verdict = "MED" if r["is_medicine"] else "REJ"
    # sanitise for filename
    safe_raw = "".join(c if c.isalnum() else "-" for c in raw)
    safe_snap = "".join(c if c.isalnum() else "-" for c in snap)
    fname = f"line{r['index']:02d}_{verdict}_RAW-{safe_raw}_SNAP-{safe_snap}_DIST-{r['match_distance']}.jpg"
    cv2.imwrite(str(OUT/fname), r["crop"])
    print(f"  {fname}")

print(f"\n>>> Open the folder '{OUT}/' and look at each image.")
print(">>> KEY QUESTION: does each crop's actual content match its RAW-xxx label?")
print(">>>   - If a crop of a DATE/CREDENTIAL has a RAW medicine name -> TrOCR-cannot-reject problem")
print(">>>   - If crops are garbled/duplicated/wrong regions -> segmentation problem")
