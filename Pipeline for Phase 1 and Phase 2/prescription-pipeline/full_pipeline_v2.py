"""
full_pipeline_v2.py
===================
Complete prescription -> medicine-names workflow WITH a medicine-validation gate.

    full image
      -> segmentation  (crops, may include non-medicine regions)
      -> HTR recognition (augmented TrOCR)
      -> formulary normalisation (snap to nearest valid drug name)
      -> VALIDATION GATE: was it a confident formulary match?  keep / flag
      -> output: medicines only (non-medicine regions flagged, not shown)

The gate uses the formulary as a VALIDATOR: a crop is accepted as a medicine
only if its recognised text matches a real drug name closely enough. Credentials,
dates, and professions ("SLMC Reg No", "Diploma in Allergy") land far from every
drug name and are rejected.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image
import torch
import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from segmentation_pipeline import PrescriptionSegmenter


def _edit_distance(a, b):
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        cur=[i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]


class PrescriptionPipeline:
    def __init__(self, craft_model_path, craft_repo_dir, trocr_ckpt,
                 formulary_csv, train_csv,
                 tau=0.4,            # normalisation snap threshold
                 accept_tau=0.35,    # VALIDATION threshold: <= this = accept as medicine
                 min_len=3,          # reject very short junk
                 device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                  else "cuda" if torch.cuda.is_available() else "cpu")
        self.tau = tau
        self.accept_tau = accept_tau
        self.min_len = min_len

        self.seg = PrescriptionSegmenter(craft_model_path=craft_model_path,
                                         craft_repo_dir=craft_repo_dir, device=self.device)
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained(trocr_ckpt).to(self.device).eval()

        train_names = set(pd.read_csv(train_csv)["medicine_name"].astype(str).str.strip().str.lower())
        names = set(train_names)
        if Path(formulary_csv).exists():
            ext = pd.read_csv(formulary_csv)
            col = "drug_name" if "drug_name" in ext.columns else ext.columns[0]
            names |= (set(ext[col].astype(str).str.strip().str.lower()) - {""})
        self.formulary = sorted(names)
        self._by_len = defaultdict(list)
        for w in self.formulary:
            self._by_len[len(w)].append(w)

    def _nearest(self, word):
        """Return (nearest_formulary_name, normalised_distance)."""
        if not word:
            return None, 1.0
        if word in self._by_len.get(len(word), ()):
            return word, 0.0
        best, bd = None, 10**9
        for L in range(len(word)-3, len(word)+4):
            for e in self._by_len.get(L, ()):
                d = _edit_distance(word, e)
                if d < bd: best, bd = e, d
                if bd == 1: break
        return best, (bd / max(len(word), 1) if best is not None else 1.0)

    def _recognise(self, crop_bgr):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pv = self.processor(pil, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            ids = self.model.generate(pv, max_new_tokens=24)
        return self.processor.decode(ids[0], skip_special_tokens=True).strip().lower()

    # ===================================================================
    # PUBLIC ENTRY POINT
    # ===================================================================
    def run(self, image_path, save_crops_dir=None):
        seg_result = self.seg.segment(image_path, save_dir=save_crops_dir)
        all_lines, medicines = [], []
        for line in seg_result["lines"]:
            raw = self._recognise(line["image"])
            nearest, dist = self._nearest(raw)

            # VALIDATION GATE
            is_medicine = (len(raw) >= self.min_len) and (dist <= self.accept_tau)
            final_name = nearest if (is_medicine and dist <= self.tau) else raw

            record = {"index": line["index"], "raw_recognition": raw,
                      "medicine_name": final_name if is_medicine else None,
                      "match_distance": round(dist, 3),
                      "is_medicine": is_medicine,
                      "bbox": line["bbox"], "crop": line["image"]}
            all_lines.append(record)
            if is_medicine:
                medicines.append(record)

        return {"source": image_path,
                "total_regions": len(all_lines),
                "medicines_found": len(medicines),
                "rejected": len(all_lines) - len(medicines),
                "medicines": medicines,     # <-- show these in the UI/demo
                "all_regions": all_lines}   # <-- full detail (incl. rejected), for honesty/debug


if __name__ == "__main__":
    pipe = PrescriptionPipeline(
        craft_model_path="models/craft_mlt_25k.pth",
        craft_repo_dir="libs/CRAFT-pytorch",
        trocr_ckpt="checkpoints/trocr_augmented/best",
        formulary_csv="data/formulary/drug_names.csv",
        train_csv="data/pharmacy_lk/splits/train.csv",
    )
    res = pipe.run("data/test_images/176805_1.Jpg")
    print(f"{res['medicines_found']} medicines found "
          f"({res['rejected']} non-medicine regions filtered out)")
    print("\nMEDICINES:")
    for r in res["medicines"]:
        print(f"  line {r['index']}: {r['medicine_name']}  (dist {r['match_distance']})")
    print("\nREJECTED (non-medicine):")
    for r in res["all_regions"]:
        if not r["is_medicine"]:
            print(f"  line {r['index']}: raw='{r['raw_recognition']}' (dist {r['match_distance']}) -> rejected")
