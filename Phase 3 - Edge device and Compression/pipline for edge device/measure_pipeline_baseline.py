#!/usr/bin/env python3
"""
measure_pipeline_baseline.py
============================
Baseline 1(c): CRAFT (Phase 1) latency + END-TO-END latency per prescription,
measured through YOUR ACTUAL pipeline classes (segmentation_pipeline +
full_pipeline_v2) -- not a reconstruction.

DROP THIS FILE IN YOUR PROJECT ROOT (next to segmentation_pipeline.py and
full_pipeline_v2.py) and run it from there, in the `pipeline` conda env.

Per full prescription image it reports the distribution (median / p95) of:
  - CRAFT network forward     -> the compressible Phase-1 neural cost
  - Phase-1 segment() total   -> CRAFT net + OpenCV flatten/filter/merge/crop
  - recognition (derived)     -> = end-to-end - segment(); all TrOCR lines
  - END-TO-END run()          -> segment -> TrOCR x N lines -> normalise -> gate
  - lines per prescription    -> drives how end-to-end scales

Why the split: only the neural parts (CRAFT net, TrOCR) are compressible; the
OpenCV ops + formulary lookup are fixed cost. This is the CEILING on what
compression can save end-to-end -- the honest number for the thesis.

USAGE (from project root):
  PYTORCH_ENABLE_MPS_FALLBACK=1 python measure_pipeline_baseline.py \
      --images data/test_images --max-images 20 --inner 2 --warmup 2

Model/formulary path defaults match your full_pipeline_v2 __main__, so the
bare command usually works.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoids the OMP abort

import argparse
import statistics
import time
from glob import glob

import torch

from full_pipeline_v2 import PrescriptionPipeline   # your real pipeline

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def summarize(name, values):
    vals = [v for v in values if v is not None]
    if not vals:
        print(f"  [{name}]  no samples")
        return None
    v = sorted(vals)
    n = len(v)
    stats = dict(mean=statistics.mean(v), median=statistics.median(v),
                 p95=v[min(n - 1, round(0.95 * n) - 1)],
                 std=statistics.pstdev(v), mn=v[0], mx=v[-1])
    print(f"  [{name}]  n={n}")
    print(f"     mean {stats['mean']:.1f} | median {stats['median']:.1f} "
          f"| p95 {stats['p95']:.1f} | std {stats['std']:.1f} "
          f"| min {stats['mn']:.1f} | max {stats['mx']:.1f}   (ms)")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/test_images",
                    help="folder of FULL prescription images")
    ap.add_argument("--craft", default="models/craft_mlt_25k.pth")
    ap.add_argument("--craft-repo", default="libs/CRAFT-pytorch")
    ap.add_argument("--trocr", default="checkpoints/trocr_augmented/best")
    ap.add_argument("--formulary", default="data/formulary/drug_names.csv")
    ap.add_argument("--train-csv", default="data/pharmacy_lk/splits/train.csv")
    ap.add_argument("--max-images", type=int, default=20)
    ap.add_argument("--inner", type=int, default=2,
                    help="reps per image for the segment() timing (median taken)")
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    img_paths = sorted(p for p in glob(os.path.join(args.images, "*"))
                       if p.lower().endswith(IMG_EXT))[:args.max_images]
    if not img_paths:
        ap.error(f"no images in --images '{args.images}' "
                 f"(point this at FULL prescription photos, not line crops)")
    print(f"images: {len(img_paths)} full prescriptions from {args.images}")

    pipe = PrescriptionPipeline(
        craft_model_path=args.craft, craft_repo_dir=args.craft_repo,
        trocr_ckpt=args.trocr, formulary_csv=args.formulary,
        train_csv=args.train_csv)
    device = pipe.device
    print(f"device: {device}\n")

    # ---- warmup ----
    for _ in range(args.warmup):
        pipe.seg.segment(img_paths[0])
        pipe.run(img_paths[0])
    sync(device)

    # ---- CRAFT-net forward timing via hooks (active only in segment loop) ----
    craft_state = {"start": None, "samples": []}

    def _pre(module, inp):
        sync(device)
        craft_state["start"] = time.perf_counter()

    def _post(module, inp, out):
        sync(device)
        craft_state["samples"].append((time.perf_counter() - craft_state["start"]) * 1000.0)

    h1 = pipe.seg.net.register_forward_pre_hook(_pre)
    h2 = pipe.seg.net.register_forward_hook(_post)

    per_image = []
    for path in img_paths:
        seg_reps, craft_reps = [], []
        for _ in range(args.inner):
            craft_state["samples"].clear()
            t0 = time.perf_counter()
            pipe.seg.segment(path)
            sync(device)
            seg_reps.append((time.perf_counter() - t0) * 1000.0)
            if craft_state["samples"]:
                craft_reps.append(statistics.median(craft_state["samples"]))
        per_image.append(dict(
            path=path,
            segment_ms=statistics.median(seg_reps),
            craft_ms=statistics.median(craft_reps) if craft_reps else None))

    h1.remove()
    h2.remove()

    # ---- end-to-end timing (hooks removed) ----
    for rec in per_image:
        t0 = time.perf_counter()
        result = pipe.run(rec["path"])
        sync(device)
        rec["e2e_ms"] = (time.perf_counter() - t0) * 1000.0
        rec["n_lines"] = result["total_regions"]
        rec["recognition_ms"] = max(0.0, rec["e2e_ms"] - rec["segment_ms"])

    # ---- report ----
    print(f"Per-prescription latency (across {len(per_image)} images)")
    summarize("CRAFT network forward", [r["craft_ms"] for r in per_image])
    summarize("Phase-1 segment() total", [r["segment_ms"] for r in per_image])
    summarize("Recognition (derived, all lines)", [r["recognition_ms"] for r in per_image])
    summarize("END-TO-END run()", [r["e2e_ms"] for r in per_image])

    lines = [r["n_lines"] for r in per_image]
    print(f"\n  lines/prescription: mean {statistics.mean(lines):.1f} "
          f"| median {statistics.median(lines):.0f} "
          f"| min {min(lines)} | max {max(lines)}")

    print("\nNotes for the thesis:")
    print(" - Quote median + p95 of END-TO-END run() as the per-prescription baseline.")
    print(" - CRAFT-net vs segment() = neural (compressible) vs OpenCV (fixed) split.")
    print(" - Recognition scales with lines/prescription; report it with the count.")
    print(" - Recognition here uses full_pipeline_v2's max_new_tokens=24.")


if __name__ == "__main__":
    main()
