#!/usr/bin/env python3
"""
measure_onnx_cpu.py -- Step 2 close-out: PyTorch-CPU vs ONNX-CPU (both FP32).

Same hardware (CPU), same precision (FP32) -- only the RUNTIME differs. This is
the honest control for "what does moving to ONNX Runtime buy us?" It is NOT
comparable to the earlier 44 ms/line MPS number (different device); that stays
as a separate row. CPU numbers here are the better preview of the Pi (CPU-only).

Per cropped line (TrOCR generate) and per full image (CRAFT forward) it prints
median/p95 for PyTorch-CPU and ONNX-CPU, plus the speed ratio.

Run from PROJECT ROOT, `pipeline` env, after export_onnx.py + verify_onnx.py.

USAGE:
  python measure_onnx_cpu.py \
      --lines data/cropped_test_images --n-lines 20 \
      --image data/test_images --n-images 6 \
      --max-new-tokens 24 --inner 1 --warmup 2
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import os.path as osp
import statistics
import time
from glob import glob

import torch
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
torch.set_num_threads(os.cpu_count() or 4)


def stats_ms(values):
    v = sorted(values)
    n = len(v)
    return dict(median=statistics.median(v),
                p95=v[min(n - 1, round(0.95 * n) - 1)],
                mean=statistics.mean(v), std=statistics.pstdev(v), n=n)


def show(label, s):
    print(f"  [{label}]  n={s['n']}  median {s['median']:.1f} | p95 {s['p95']:.1f} "
          f"| mean {s['mean']:.1f} | std {s['std']:.1f}   (ms)")


def time_across(fn, inputs, warmup, inner, label):
    for _ in range(warmup):
        fn(inputs[0])
    per = []
    for x in inputs:
        reps = []
        for _ in range(inner):
            t0 = time.perf_counter()
            fn(x)
            reps.append((time.perf_counter() - t0) * 1000.0)
        per.append(statistics.median(reps))
    s = stats_ms(per)
    show(label, s)
    return s


def list_images(path, n):
    if osp.isdir(path):
        ps = [p for p in sorted(glob(osp.join(path, "*")))
              if p.lower().endswith(IMG_EXT)]
    else:
        ps = [path]
    return ps[:n]


def ratio_line(s_pt, s_ort):
    r = s_pt["median"] / s_ort["median"]
    word = "faster" if s_ort["median"] < s_pt["median"] else "slower"
    print(f"  -> ONNX-CPU median is {r:.2f}x {word} than PyTorch-CPU\n")


def bench_trocr(args):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from optimum.onnxruntime import ORTModelForVision2Seq

    print("TrOCR  (per cropped line, CPU, FP32)")
    try:
        proc = TrOCRProcessor.from_pretrained(args.trocr_onnx)
    except Exception:
        proc = TrOCRProcessor.from_pretrained(args.processor)
    pt = VisionEncoderDecoderModel.from_pretrained(args.trocr).to("cpu").eval()
    ort = ORTModelForVision2Seq.from_pretrained(args.trocr_onnx)

    paths = list_images(args.lines, args.n_lines)
    if not paths:
        print("  no line crops found\n")
        return
    pvs = [proc(images=Image.open(p).convert("RGB"),
                return_tensors="pt").pixel_values for p in paths]
    mnt = args.max_new_tokens

    def pt_fn(pv):
        with torch.no_grad():
            pt.generate(pv, max_new_tokens=mnt)

    def ort_fn(pv):
        ort.generate(pv, max_new_tokens=mnt)

    s_pt = time_across(pt_fn, pvs, args.warmup, args.inner, "PyTorch-CPU")
    s_ort = time_across(ort_fn, pvs, args.warmup, args.inner, "ONNX-CPU")
    ratio_line(s_pt, s_ort)


def craft_input(seg, image_path):
    import cv2
    raw = cv2.imread(image_path)
    flat = seg._flatten_paper(raw)
    flat_rgb = cv2.cvtColor(flat, cv2.COLOR_BGR2RGB)
    p = seg.craft_params
    img_resized, _, _ = seg._imgproc.resize_aspect_ratio(
        flat_rgb, p["canvas_size"], interpolation=cv2.INTER_LINEAR,
        mag_ratio=p["mag_ratio"])
    x = seg._imgproc.normalizeMeanVariance(img_resized)
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()


def bench_craft(args):
    import onnxruntime as ortrt
    from segmentation_pipeline import PrescriptionSegmenter

    print("CRAFT  (per full image, forward only, CPU, FP32)")
    seg = PrescriptionSegmenter(craft_model_path=args.craft,
                                craft_repo_dir=args.craft_repo, device="cpu")
    paths = list_images(args.image, args.n_images)
    if not paths:
        print("  no full images found\n")
        return
    xs = [craft_input(seg, p) for p in paths]

    def pt_fn(x):
        with torch.no_grad():
            seg.net(x)

    sess = ortrt.InferenceSession(args.craft_onnx,
                                  providers=["CPUExecutionProvider"])

    def ort_fn(x):
        sess.run(None, {"input": x.numpy()})

    s_pt = time_across(pt_fn, xs, args.warmup, args.inner, "PyTorch-CPU")
    s_ort = time_across(ort_fn, xs, args.warmup, args.inner, "ONNX-CPU")
    ratio_line(s_pt, s_ort)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trocr", default="checkpoints/trocr_augmented/best")
    ap.add_argument("--trocr-onnx", default="onnx_models/trocr")
    ap.add_argument("--processor", default="microsoft/trocr-small-handwritten")
    ap.add_argument("--lines", default="data/cropped_test_images")
    ap.add_argument("--n-lines", type=int, default=20)
    ap.add_argument("--craft", default="models/craft_mlt_25k.pth")
    ap.add_argument("--craft-repo", default="libs/CRAFT-pytorch")
    ap.add_argument("--craft-onnx", default="onnx_models/craft/craft.onnx")
    ap.add_argument("--image", default="data/test_images")
    ap.add_argument("--n-images", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--inner", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--only", choices=["trocr", "craft"])
    args = ap.parse_args()

    print(f"threads: {torch.get_num_threads()}  (CPU timing, FP32)\n")
    if args.only != "craft":
        bench_trocr(args)
    if args.only != "trocr":
        bench_craft(args)
    print("Reminder: these are CPU/FP32 numbers. The honest comparison is")
    print("PyTorch-CPU vs ONNX-CPU above. The 44 ms/line MPS figure is a")
    print("different device and stays as its own row in the thesis.")


if __name__ == "__main__":
    main()
