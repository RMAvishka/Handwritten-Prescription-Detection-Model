#!/usr/bin/env python3
"""
eval_trocr_accuracy.py -- Step 3a accuracy: does INT8 TrOCR still hit 0.721?

Mirrors notebook 22 EXACTLY (edit_distance, nearest/snap with tau=0.4, formulary
= train lexicon UNION external drug_names) but runs each TrOCR variant and prints
EM/CER side by side, plus the seen/unseen split, so the INT8-vs-FP32 comparison
is airtight.

Variants timed for accuracy:
  - PyTorch FP32   (your original recogniser; should reproduce 0.6549 / 0.7206)
  - ONNX FP32      (exported model)
  - ONNX INT8      (quantized model)            <-- the number we care about

Two EM numbers per variant:
  RAW       = model output vs ground truth          (pure quantization effect)
  +FORMULARY= after snap to nearest drug name        (what the user experiences)

Run from PROJECT ROOT, `pipeline` env, after quantize_trocr.py.
USAGE:
  python eval_trocr_accuracy.py            # all three variants
  python eval_trocr_accuracy.py --only int8
  python eval_trocr_accuracy.py --limit 100   # quick subset while iterating
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

# ----- notebook-identical metric + lexicon machinery -----------------------
def edit_distance(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def metrics(preds, refs):
    tot = sum(edit_distance(p, r) for p, r in zip(preds, refs))
    chars = sum(len(r) for r in refs)
    em = sum(p == r for p, r in zip(preds, refs))
    return {"CER": tot / max(chars, 1), "EM": em / len(refs), "n": len(refs)}


def build_index(names):
    bl = defaultdict(list)
    for w in names:
        bl[len(w)].append(w)
    return set(names), bl


def nearest(word, by_len, gap=3):
    if not word:
        return None, 10 ** 9
    if word in by_len.get(len(word), ()):
        return word, 0
    best, bd = None, 10 ** 9
    for L in range(len(word) - gap, len(word) + gap + 1):
        for e in by_len.get(L, ()):
            d = edit_distance(word, e)
            if d < bd:
                best, bd = e, d
            if bd == 1:
                return best, bd
    return best, bd


def snap(word, by_len, tau=0.4):
    e, d = nearest(word, by_len)
    return e if (e is not None and d / max(len(word), 1) <= tau) else word


# ----- model runners -------------------------------------------------------
def run_pytorch_fp32(ckpt, processor_src, pils, device, mnt):
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    try:
        proc = TrOCRProcessor.from_pretrained(ckpt)
    except Exception:
        proc = TrOCRProcessor.from_pretrained(processor_src)
    model = VisionEncoderDecoderModel.from_pretrained(ckpt).to(device).eval()
    out = []
    for pil in pils:
        pv = proc(pil, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(pv, max_new_tokens=mnt)
        out.append(proc.decode(ids[0], skip_special_tokens=True).strip().lower())
    return out


def run_onnx(onnx_dir, processor_src, pils, mnt):
    from transformers import TrOCRProcessor
    from optimum.onnxruntime import ORTModelForVision2Seq
    try:
        proc = TrOCRProcessor.from_pretrained(onnx_dir)
    except Exception:
        proc = TrOCRProcessor.from_pretrained(processor_src)
    model = ORTModelForVision2Seq.from_pretrained(onnx_dir)
    out = []
    for pil in pils:
        pv = proc(pil, return_tensors="pt").pixel_values
        ids = model.generate(pv, max_new_tokens=mnt)
        out.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip().lower())
    return out


def report(name, raw, refs, train_idx, ext_idx, train_set):
    pred_ext = [snap(p, ext_idx) for p in raw]
    m_raw = metrics(raw, refs)
    m_ext = metrics(pred_ext, refs)
    seen_mask = [r in train_set for r in refs]

    def split_em(preds):
        s = [(p, r) for p, r, se in zip(preds, refs, seen_mask) if se]
        u = [(p, r) for p, r, se in zip(preds, refs, seen_mask) if not se]
        se = metrics([p for p, _ in s], [r for _, r in s])["EM"] if s else 0
        ue = metrics([p for p, _ in u], [r for _, r in u])["EM"] if u else 0
        return se, ue, len(s), len(u)

    se, ue, ns, nu = split_em(pred_ext)
    print(f"\n=== {name} ===")
    print(f"  RAW        : EM {m_raw['EM']:.4f} | CER {m_raw['CER']:.4f}")
    print(f"  +FORMULARY : EM {m_ext['EM']:.4f} | CER {m_ext['CER']:.4f}")
    print(f"  seen {se:.3f} (n={ns}) | unseen {ue:.3f} (n={nu})")
    return {"variant": name, "EM_raw": m_raw["EM"], "EM_formulary": m_ext["EM"],
            "CER_raw": m_raw["CER"], "n": m_raw["n"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/pharmacy_lk")
    ap.add_argument("--test-csv", default="data/pharmacy_lk/splits/test.csv")
    ap.add_argument("--train-csv", default="data/pharmacy_lk/splits/train.csv")
    ap.add_argument("--formulary", default="data/formulary/drug_names.csv")
    ap.add_argument("--img-col", default="image_filename")
    ap.add_argument("--label-col", default="medicine_name")
    ap.add_argument("--ckpt", default="checkpoints/trocr_augmented/best")
    ap.add_argument("--onnx-fp32", default="onnx_models/trocr")
    ap.add_argument("--onnx-int8", default="onnx_models/trocr_int8")
    ap.add_argument("--processor", default="microsoft/trocr-small-handwritten")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="0 = all test rows")
    ap.add_argument("--only", choices=["pt", "fp32", "int8"],
                    help="run a single variant")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    img_col, label_col = args.img_col, args.label_col

    # build the SAME formulary the notebook used: train lexicon UNION external
    train_names = sorted(set(pd.read_csv(args.train_csv)[label_col]
                             .astype(str).str.strip().str.lower()))
    ext = pd.read_csv(args.formulary)
    col = "drug_name" if "drug_name" in ext.columns else ext.columns[0]
    external = sorted(set(ext[col].astype(str).str.strip().str.lower()) - {""}
                      | set(train_names))
    train_set, train_idx = build_index(train_names)
    ext_set, ext_idx = build_index(external)
    print(f"train lexicon: {len(train_names)} | external formulary: {len(external)}")

    # load the SAME test rows (image + ground truth)
    df = pd.read_csv(args.test_csv).dropna(subset=[label_col])
    pils, refs = [], []
    for _, r in df.iterrows():
        p = data_root / "images" / str(r[img_col])
        if not p.exists():
            continue
        pils.append(Image.open(p).convert("RGB"))
        refs.append(str(r[label_col]).strip().lower())
        if args.limit and len(pils) >= args.limit:
            break
    print(f"test images loaded: {len(pils)}")

    import torch
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    if args.only in (None, "pt"):
        raw = run_pytorch_fp32(args.ckpt, args.processor, pils, device, args.max_new_tokens)
        rows.append(report("PyTorch FP32", raw, refs, train_idx, ext_idx, train_set))
    if args.only in (None, "fp32"):
        raw = run_onnx(args.onnx_fp32, args.processor, pils, args.max_new_tokens)
        rows.append(report("ONNX FP32", raw, refs, train_idx, ext_idx, train_set))
    if args.only in (None, "int8"):
        raw = run_onnx(args.onnx_int8, args.processor, pils, args.max_new_tokens)
        rows.append(report("ONNX INT8", raw, refs, train_idx, ext_idx, train_set))

    if len(rows) > 1:
        print("\n--- SUMMARY (EM) ---")
        print(f"  {'variant':14s}  {'EM_raw':>8s}  {'EM_formulary':>12s}")
        for r in rows:
            print(f"  {r['variant']:14s}  {r['EM_raw']:.4f}    {r['EM_formulary']:.4f}")
        base = rows[0]["EM_formulary"]
        for r in rows[1:]:
            d = r["EM_formulary"] - base
            print(f"  {r['variant']} vs {rows[0]['variant']} (+formulary): "
                  f"{d:+.4f} EM")
    pd.DataFrame(rows).to_csv("trocr_quant_accuracy.csv", index=False)
    print("\nsaved -> trocr_quant_accuracy.csv")


if __name__ == "__main__":
    main()
