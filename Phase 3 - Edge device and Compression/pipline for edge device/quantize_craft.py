#!/usr/bin/env python3
"""
quantize_craft.py -- Step 3b: INT8 STATIC quantization of CRAFT (PyTorch).

WHY STATIC (not dynamic): CRAFT is a CNN (Conv2d). PyTorch DYNAMIC quantization
only handles Linear/RNN, so it would compress ~nothing here. STATIC quantization
(calibrate activation ranges on real images, then convert) is the only PyTorch
path that actually compresses a conv-net -> ~4x smaller, and on ARM (Pi) qnnpack
can give some conv speed-up.

Flow:
  load CRAFT (via your PrescriptionSegmenter)
   -> FX prepare with qnnpack qconfig
   -> calibrate on N real flattened prescription images
   -> convert to INT8
   -> measure SIZE, SPEED (CPU), and DETECTION faithfulness (lines vs FP32)

Honest expectations:
  SIZE ~4x smaller (near-guaranteed). SPEED uncertain (upsampling stays FP32;
  Pi-4 A72 lacks fast INT8 ops). DETECTION measured here; if it diverges badly,
  the honest fallback is to keep CRAFT FP32 and report the attempt + size.

Run from PROJECT ROOT, `pipeline` env.
USAGE:
  python quantize_craft.py --calib data/test_images --n-calib 30 \
      --eval data/test_images --n-eval 6
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import copy
import os.path as osp
import statistics
import time
from glob import glob

import torch
import torch.nn as nn

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
torch.backends.quantized.engine = "qnnpack"   # ARM backend (Pi + Apple Silicon)


def list_images(path, n):
    if osp.isdir(path):
        ps = [p for p in sorted(glob(osp.join(path, "*")))
              if p.lower().endswith(IMG_EXT)]
    else:
        ps = [path]
    return ps[:n]


def craft_input(seg, path):
    import cv2
    raw = cv2.imread(path)
    flat = seg._flatten_paper(raw)
    rgb = cv2.cvtColor(flat, cv2.COLOR_BGR2RGB)
    p = seg.craft_params
    img, _, _ = seg._imgproc.resize_aspect_ratio(
        rgb, p["canvas_size"], interpolation=cv2.INTER_LINEAR,
        mag_ratio=p["mag_ratio"])
    x = seg._imgproc.normalizeMeanVariance(img)
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()


class DequantWrap(nn.Module):
    """Force float outputs so the existing pipeline (.numpy()) keeps working."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        y, feat = out if isinstance(out, tuple) else (out, None)
        if hasattr(y, "is_quantized") and y.is_quantized:
            y = y.dequantize()
        if feat is not None and hasattr(feat, "is_quantized") and feat.is_quantized:
            feat = feat.dequantize()
        return y, feat


def state_size_mb(model, tag):
    path = f"/tmp/_craft_{tag}.pt"
    torch.save(model.state_dict(), path)
    mb = osp.getsize(path) / 1024 / 1024
    os.remove(path)
    return mb


def bench(net, xs, warmup, label):
    net.eval()
    with torch.no_grad():
        for _ in range(warmup):
            net(xs[0])
        per = []
        for x in xs:
            t0 = time.perf_counter()
            net(x)
            per.append((time.perf_counter() - t0) * 1000.0)
    per.sort()
    n = len(per)
    med = statistics.median(per)
    print(f"  [{label}] median {med:.1f} | "
          f"p95 {per[min(n - 1, round(0.95 * n) - 1)]:.1f}  (ms)")
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--craft", default="models/craft_mlt_25k.pth")
    ap.add_argument("--craft-repo", default="libs/CRAFT-pytorch")
    ap.add_argument("--calib", default="data/test_images")
    ap.add_argument("--n-calib", type=int, default=30)
    ap.add_argument("--eval", default="data/test_images")
    ap.add_argument("--n-eval", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="models/craft_int8.pt")
    args = ap.parse_args()

    from segmentation_pipeline import PrescriptionSegmenter
    seg = PrescriptionSegmenter(craft_model_path=args.craft,
                                craft_repo_dir=args.craft_repo, device="cpu")
    fp32 = seg.net.cpu().eval()

    calib_paths = list_images(args.calib, args.n_calib)
    eval_paths = list_images(args.eval, args.n_eval)
    if not calib_paths:
        raise SystemExit("no calibration images found at --calib")
    print(f"calibration images: {len(calib_paths)} | eval images: {len(eval_paths)}")
    if len(calib_paths) < 20:
        print("  (note: few calibration images -> activation ranges are rough; "
              "more full prescriptions would improve INT8 accuracy)")
    calib_xs = [craft_input(seg, p) for p in calib_paths]
    eval_xs = [craft_input(seg, p) for p in eval_paths]
    example = (calib_xs[0],)

    # ---- FX static quantization ----
    try:
        from torch.ao.quantization import get_default_qconfig, QConfigMapping
        from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

        qmap = QConfigMapping().set_global(get_default_qconfig("qnnpack"))
        print("\npreparing (FX) + calibrating ...")
        prepared = prepare_fx(copy.deepcopy(fp32), qmap, example)
        with torch.no_grad():
            for x in calib_xs:
                prepared(x)
        quantized = convert_fx(prepared)
        qnet = DequantWrap(quantized).eval()
        # sanity: one forward must run
        with torch.no_grad():
            _ = qnet(eval_xs[0])
    except Exception as e:
        print(f"\n[FX static quant FAILED] {type(e).__name__}: {e}")
        print("This is the 'CNN resisted quantization' branch. Paste this error")
        print("and we'll either switch to eager-mode quant or keep CRAFT FP32")
        print("(a valid, reportable outcome). Stopping here.")
        return

    # ---- size ----
    torch.save(quantized.state_dict(), args.out)
    s_fp = state_size_mb(fp32, "fp32")
    s_q = osp.getsize(args.out) / 1024 / 1024
    print(f"\nSize: FP32 {s_fp:.1f} MB -> INT8 {s_q:.1f} MB "
          f"({s_fp / max(s_q, 1e-9):.2f}x smaller)   saved -> {args.out}")

    # ---- speed ----
    print("\nSpeed (CRAFT forward, CPU):")
    md_fp = bench(fp32, eval_xs, args.warmup, "FP32")
    md_q = bench(qnet, eval_xs, args.warmup, "INT8 (qnnpack)")
    word = "faster" if md_q < md_fp else "slower"
    print(f"  -> INT8 median is {md_fp / md_q:.2f}x {word} than FP32")

    # ---- detection faithfulness: lines found, FP32 vs INT8 ----
    print("\nDetection faithfulness (lines detected per prescription):")
    fp_counts = [seg.segment(p)["count"] for p in eval_paths]   # seg.net is fp32
    seg.net = qnet                                              # swap in INT8
    q_counts = [seg.segment(p)["count"] for p in eval_paths]
    print(f"  FP32 lines: {fp_counts}")
    print(f"  INT8 lines: {q_counts}")
    same = sum(a == b for a, b in zip(fp_counts, q_counts))
    print(f"  same line-count on {same}/{len(eval_paths)} prescriptions")

    print("\nRead: INT8 counts close to FP32 => detection preserved (good). Big")
    print("divergence => quantization hurt detection; we'd keep CRAFT FP32 and")
    print("report the size result + the honest accuracy/robustness trade-off.")


if __name__ == "__main__":
    main()
