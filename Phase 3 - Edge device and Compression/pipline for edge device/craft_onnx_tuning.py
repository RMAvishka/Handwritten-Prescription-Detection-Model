#!/usr/bin/env python3
"""
craft_onnx_tuning.py -- why is CRAFT-ONNX slower on CPU, and is it fixable?

FP32 CRAFT was ~2.5x SLOWER under ONNX Runtime than PyTorch on CPU. Before we
quantize, find out whether that's (a) an artifact of the new 'dynamo' exporter,
or (b) intrinsic. Compares CRAFT forward latency for:
   1. PyTorch-CPU                          (reference)
   2. ONNX dynamo export, DEFAULT session
   3. ONNX dynamo export, TUNED session    (graph_opt=ALL, all threads)
   4. ONNX LEGACY export, tuned session    (if dynamo=False export still works)

Whichever ONNX row is fastest is the CRAFT artifact we carry into Step 3.

Run from PROJECT ROOT, `pipeline` env.
USAGE: python craft_onnx_tuning.py --image data/test_images --n-images 6
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import os.path as osp
import statistics
import time
from glob import glob

import torch

torch.set_num_threads(os.cpu_count() or 4)
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def craft_input(seg, path):
    import cv2
    raw = cv2.imread(path)
    flat = seg._flatten_paper(raw)
    rgb = cv2.cvtColor(flat, cv2.COLOR_BGR2RGB)
    pr = seg.craft_params
    img, _, _ = seg._imgproc.resize_aspect_ratio(
        rgb, pr["canvas_size"], interpolation=cv2.INTER_LINEAR,
        mag_ratio=pr["mag_ratio"])
    x = seg._imgproc.normalizeMeanVariance(img)
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()


def tstats(vals):
    v = sorted(vals)
    n = len(v)
    return dict(median=statistics.median(v),
                p95=v[min(n - 1, round(0.95 * n) - 1)],
                mean=statistics.mean(v), n=n)


def bench(fn, xs, warmup, label):
    for _ in range(warmup):
        fn(xs[0])
    per = []
    for x in xs:
        t0 = time.perf_counter()
        fn(x)
        per.append((time.perf_counter() - t0) * 1000.0)
    s = tstats(per)
    print(f"  [{label}]  n={s['n']}  median {s['median']:.1f} | "
          f"p95 {s['p95']:.1f} | mean {s['mean']:.1f}  (ms)")
    return s


def make_session(path, tuned):
    import onnxruntime as ort
    if not tuned:
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = os.cpu_count() or 4
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--craft", default="models/craft_mlt_25k.pth")
    ap.add_argument("--craft-repo", default="libs/CRAFT-pytorch")
    ap.add_argument("--dynamo-onnx", default="onnx_models/craft/craft.onnx")
    ap.add_argument("--legacy-onnx", default="onnx_models/craft/craft_legacy.onnx")
    ap.add_argument("--image", default="data/test_images")
    ap.add_argument("--n-images", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=768)
    args = ap.parse_args()

    from segmentation_pipeline import PrescriptionSegmenter
    seg = PrescriptionSegmenter(craft_model_path=args.craft,
                                craft_repo_dir=args.craft_repo, device="cpu")
    if osp.isdir(args.image):
        paths = [p for p in sorted(glob(osp.join(args.image, "*")))
                 if p.lower().endswith(IMG_EXT)][:args.n_images]
    else:
        paths = [args.image]
    xs = [craft_input(seg, p) for p in paths]
    print(f"threads: {torch.get_num_threads()} | images: {len(xs)}\n")

    def pt_fn(x):
        with torch.no_grad():
            seg.net(x)
    bench(pt_fn, xs, args.warmup, "PyTorch-CPU")

    sess_d = make_session(args.dynamo_onnx, tuned=False)
    bench(lambda x: sess_d.run(None, {"input": x.numpy()}), xs, args.warmup,
          "ONNX dynamo, default")
    sess_dt = make_session(args.dynamo_onnx, tuned=True)
    bench(lambda x: sess_dt.run(None, {"input": x.numpy()}), xs, args.warmup,
          "ONNX dynamo, tuned")

    try:
        net = seg.net.cpu().eval()
        dummy = torch.randn(1, 3, args.height, args.width)
        print("\n[legacy] exporting CRAFT with dynamo=False ...")
        torch.onnx.export(
            net, dummy, args.legacy_onnx, dynamo=False,
            input_names=["input"], output_names=["score", "feature"],
            dynamic_axes={"input": {0: "b", 2: "h", 3: "w"},
                          "score": {0: "b", 1: "ho", 2: "wo"}},
            opset_version=14, do_constant_folding=True)
        sess_l = make_session(args.legacy_onnx, tuned=True)
        bench(lambda x: sess_l.run(None, {"input": x.numpy()}), xs, args.warmup,
              "ONNX legacy, tuned")
    except Exception as e:
        print(f"[legacy] not available on this torch ({type(e).__name__}: {e})")
        print("        -> we'll use the dynamo export.")

    print("\nTakeaway: the fastest ONNX row is the CRAFT artifact we quantize in")
    print("Step 3. If every ONNX row stays slower than PyTorch-CPU, that's a")
    print("real finding (ONNX helps the transformer, not this CNN) and INT8")
    print("will re-decide it anyway.")


if __name__ == "__main__":
    main()
