"""
diagnose_pipeline.py — shows EXACTLY what each crop produces, to find where fake names enter.
For every segmented crop it prints:
   crop index | RAW HTR output | nearest formulary name | distance | accepted?
Run this instead of the normal pipeline to see the full picture.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from full_pipeline_v2 import PrescriptionPipeline

pipe = PrescriptionPipeline(
    craft_model_path="models/craft_mlt_25k.pth",
    craft_repo_dir="libs/CRAFT-pytorch",
    trocr_ckpt="checkpoints/trocr_augmented/best",
    formulary_csv="data/formulary/drug_names.csv",
    train_csv="data/pharmacy_lk/splits/train.csv",
)

IMAGE = "data/test_images/176805_1.Jpg"   
res = pipe.run(IMAGE)

print("\n" + "="*78)
print(f"{'idx':<5}{'RAW HTR output':<24}{'-> snapped name':<22}{'dist':<8}{'verdict'}")
print("="*78)
for r in res["all_regions"]:
    verdict = "MEDICINE" if r["is_medicine"] else "rejected"
    name = r["medicine_name"] if r["medicine_name"] else "-"
    print(f"{r['index']:<5}{r['raw_recognition']:<24}{name:<22}{r['match_distance']:<8}{verdict}")
print("="*78)
print(f"\n{res['medicines_found']} accepted as medicine, {res['rejected']} rejected")
print("\nLook for: short RAW outputs (1-4 chars) snapping to real drugs at low distance.")
print("Those are the hallucinations — printed text producing garbage that matches by luck.")
