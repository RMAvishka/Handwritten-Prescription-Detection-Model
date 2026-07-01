#!/usr/bin/env python3
"""
quantize_trocr.py -- Step 3a: INT8 DYNAMIC quantization of the TrOCR ONNX.

Dynamic INT8 targets the many Linear layers in TrOCR's attention/FFN -- the
right tool for a transformer, and it needs NO calibration data. Produces
onnx_models/trocr_int8/ and reports:
  - SIZE   : FP32 ONNX vs INT8 ONNX (expect ~4x smaller)
  - SPEED  : per-line latency, FP32-ONNX vs INT8-ONNX (CPU)
  - FAITHFULNESS PROXY: of N lines, how many decode to the SAME text as FP32.
    (The REAL number is INT8 exact-match vs your GROUND TRUTH -- the 0.721
     test set. That needs your eval script; this is just a quick sanity check.)

Run from PROJECT ROOT, `pipeline` env, after export_onnx.py.
USAGE: python quantize_trocr.py --lines data/cropped_test_images --n 20
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import os.path as osp
import shutil
import statistics
import time
from glob import glob

from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
ONNX_FILES = ["encoder_model.onnx", "decoder_model.onnx",
              "decoder_with_past_model.onnx"]


def size_mb(path):
    if osp.isfile(path):
        return osp.getsize(path) / 1024 / 1024
    t = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            t += osp.getsize(osp.join(r, f))
    return t / 1024 / 1024


def quantize(src, dst):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    os.makedirs(dst, exist_ok=True)
    for f in ONNX_FILES:
        sp = osp.join(src, f)
        if not osp.exists(sp):
            print(f"  ! missing {sp}, skipping")
            continue
        print(f"  quantizing {f} ...")
        quantize_dynamic(sp, osp.join(dst, f), weight_type=QuantType.QInt8)
    # copy aux files (config, generation_config, processor, tokenizer) so the
    # INT8 dir is a self-contained, loadable model
    for f in os.listdir(src):
        if not f.endswith(".onnx") and osp.isfile(osp.join(src, f)):
            shutil.copy(osp.join(src, f), osp.join(dst, f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", default="onnx_models/trocr")
    ap.add_argument("--int8", default="onnx_models/trocr_int8")
    ap.add_argument("--processor", default="microsoft/trocr-small-handwritten")
    ap.add_argument("--lines", default="data/cropped_test_images")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    print("Quantizing TrOCR ONNX -> INT8 (dynamic) ...")
    quantize(args.fp32, args.int8)

    s32, s8 = size_mb(args.fp32), size_mb(args.int8)
    print(f"\nSize:  FP32 {s32:.1f} MB  ->  INT8 {s8:.1f} MB  "
          f"({s32 / max(s8, 1e-9):.2f}x smaller)")
    for f in ONNX_FILES:
        a, b = osp.join(args.fp32, f), osp.join(args.int8, f)
        if osp.exists(a) and osp.exists(b):
            print(f"   {f}: {size_mb(a):.1f} -> {size_mb(b):.1f} MB")

    from transformers import TrOCRProcessor
    from optimum.onnxruntime import ORTModelForVision2Seq
    try:
        proc = TrOCRProcessor.from_pretrained(args.int8)
    except Exception:
        proc = TrOCRProcessor.from_pretrained(args.processor)
    m32 = ORTModelForVision2Seq.from_pretrained(args.fp32)
    m8 = ORTModelForVision2Seq.from_pretrained(args.int8)

    paths = [p for p in sorted(glob(osp.join(args.lines, "*")))
             if p.lower().endswith(IMG_EXT)][:args.n]
    if not paths:
        print("\n(no line crops found for timing/faithfulness)")
        return
    pvs = [proc(images=Image.open(p).convert("RGB"),
                return_tensors="pt").pixel_values for p in paths]
    mnt = args.max_new_tokens

    def timeit(model, label):
        for _ in range(args.warmup):
            model.generate(pvs[0], max_new_tokens=mnt)
        per = []
        for pv in pvs:
            t0 = time.perf_counter()
            model.generate(pv, max_new_tokens=mnt)
            per.append((time.perf_counter() - t0) * 1000.0)
        per.sort()
        n = len(per)
        med = statistics.median(per)
        print(f"  [{label}]  median {med:.1f} | "
              f"p95 {per[min(n - 1, round(0.95 * n) - 1)]:.1f}  (ms)")
        return med

    print("\nSpeed (per line, CPU):")
    md32 = timeit(m32, "FP32-ONNX")
    md8 = timeit(m8, "INT8-ONNX")
    word = "faster" if md8 < md32 else "slower"
    print(f"  -> INT8 median is {md32 / md8:.2f}x {word} than FP32-ONNX")

    print("\nFaithfulness proxy (INT8 vs FP32 text):")
    same, diffs = 0, []
    for p, pv in zip(paths, pvs):
        a = proc.batch_decode(m32.generate(pv, max_new_tokens=mnt),
                              skip_special_tokens=True)[0].strip()
        b = proc.batch_decode(m8.generate(pv, max_new_tokens=mnt),
                              skip_special_tokens=True)[0].strip()
        if a == b:
            same += 1
        else:
            diffs.append((osp.basename(p), a, b))
    print(f"  identical: {same}/{len(paths)}")
    for f, a, b in diffs[:8]:
        print(f"    {f}: fp32='{a}' | int8='{b}'")

    print("\nNote: this only checks INT8-vs-FP32 agreement. The REAL result is")
    print("INT8 exact-match vs GROUND TRUTH (your 0.721 test). Send your eval")
    print("script / labelled test set and we'll measure that next.")


if __name__ == "__main__":
    main()
