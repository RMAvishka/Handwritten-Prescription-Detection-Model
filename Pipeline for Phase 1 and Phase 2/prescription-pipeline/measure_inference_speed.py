#!/usr/bin/env python3
"""
measure_inference_speed.py
==========================
Baseline (b): wall-clock inference latency on your Mac (M4 Pro, MPS).

Measures, with warm-up + repeated timed runs and proper MPS sync:
  - TrOCR latency per cropped line  (the autoregressive decode dominates)
  - CRAFT latency per full prescription image  (detection/segmentation)
  - End-to-end latency per prescription  (CRAFT -> crop -> TrOCR x N lines)

Reports mean / median / p95 / std so you can quote a stable number and its
spread in the thesis, rather than a single noisy timing.

IMPORTANT
---------
* TrOCR section is runnable as-is against your fine-tuned HF checkpoint.
* CRAFT section is a thin wrapper you must connect to YOUR CRAFT code
  (loading + forward + box post-processing differ per repo, e.g. the
  clovaai/CRAFT-pytorch `test_net` flow). Marked with  >>> TODO <<<.
* MPS note: run with the fallback enabled so unsupported ops don't crash:
      PYTORCH_ENABLE_MPS_FALLBACK=1 python measure_inference_speed.py ...
  Ops that fall back to CPU are still legitimate to time as YOUR baseline
  on this machine; just state in the thesis that MPS fallback was enabled.

Usage
-----
  PYTORCH_ENABLE_MPS_FALLBACK=1 python measure_inference_speed.py \
      --trocr /path/to/finetuned-trocr-dir \
      --craft /path/to/craft_mlt_25k.pth \
      --image /path/to/one_prescription.jpg \
      --lines /path/to/dir_of_cropped_lines \
      --runs 30 --warmup 5
"""

import argparse
import os
import statistics
import time
from glob import glob

import torch
from PIL import Image


# --------------------------------------------------------------------------
# timing helpers
# --------------------------------------------------------------------------
def sync(device):
    """Block until queued work on the device is finished (accurate timing)."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def time_it(fn, device, runs, warmup, label):
    """Run fn() warmup+runs times; print and return timing stats (ms)."""
    for _ in range(warmup):
        fn()
    sync(device)

    times_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        sync(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    stats = {
        "mean": statistics.mean(times_ms),
        "median": statistics.median(times_ms),
        "p95": times_ms[min(len(times_ms) - 1, round(0.95 * len(times_ms)) - 1)],
        "std": statistics.pstdev(times_ms),
        "min": times_ms[0],
        "max": times_ms[-1],
    }
    print(f"  [{label}]  n={runs} (warmup {warmup})")
    print(f"     mean {stats['mean']:.1f} ms | median {stats['median']:.1f} ms "
          f"| p95 {stats['p95']:.1f} ms | std {stats['std']:.1f} ms "
          f"| min {stats['min']:.1f} | max {stats['max']:.1f}")
    return stats


def time_across_lines(infer_fn, images, device, warmup, label, inner=3):
    """
    Time `infer_fn(image)` once per DISTINCT line (median of `inner` reps each
    to suppress jitter), then report the distribution ACROSS lines. This
    captures real per-line variation (longer text -> more decoder tokens ->
    more time), which timing a single line repeatedly does not.
    """
    for _ in range(warmup):
        infer_fn(images[0])
    sync(device)

    per_line_ms = []
    for img in images:
        reps = []
        for _ in range(inner):
            t0 = time.perf_counter()
            infer_fn(img)
            sync(device)
            reps.append((time.perf_counter() - t0) * 1000.0)
        per_line_ms.append(statistics.median(reps))

    per_line_ms.sort()
    n = len(per_line_ms)
    stats = {
        "mean": statistics.mean(per_line_ms),
        "median": statistics.median(per_line_ms),
        "p95": per_line_ms[min(n - 1, round(0.95 * n) - 1)],
        "std": statistics.pstdev(per_line_ms),
        "min": per_line_ms[0],
        "max": per_line_ms[-1],
    }
    print(f"  [{label}]  across {n} distinct lines (inner {inner}, warmup {warmup})")
    print(f"     mean {stats['mean']:.1f} ms | median {stats['median']:.1f} ms "
          f"| p95 {stats['p95']:.1f} ms | std {stats['std']:.1f} ms "
          f"| min {stats['min']:.1f} | max {stats['max']:.1f}")
    return stats


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --------------------------------------------------------------------------
# TrOCR  (fully runnable)
# --------------------------------------------------------------------------
def load_trocr(path, device, processor_src="microsoft/trocr-small-handwritten"):
    """
    Model weights come from your fine-tuned checkpoint `path`.
    The processor (image preprocessor + tokenizer) is unchanged by fine-tuning,
    so if the checkpoint dir doesn't contain processor files we fall back to
    `processor_src` (the base model, or a local copy you pass via --processor).
    """
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    try:
        processor = TrOCRProcessor.from_pretrained(path)
        print(f"  processor: loaded from checkpoint ({path})")
    except (OSError, EnvironmentError):
        processor = TrOCRProcessor.from_pretrained(processor_src)
        print(f"  processor: checkpoint had none -> loaded from '{processor_src}'")

    model = VisionEncoderDecoderModel.from_pretrained(path).to(device).eval()
    return processor, model


@torch.no_grad()
def trocr_infer_one(processor, model, image, device, max_new_tokens=64):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_new_tokens=max_new_tokens)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


# --------------------------------------------------------------------------
# CRAFT  (>>> connect to your repo <<<)
# --------------------------------------------------------------------------
def load_craft(path, device):
    """
    >>> TODO <<<
    Build and return your CRAFT model from `path` (craft_mlt_25k.pth).
    For clovaai/CRAFT-pytorch this is roughly:

        from craft import CRAFT
        from collections import OrderedDict
        def strip(sd):
            return OrderedDict((k.replace('module.', ''), v) for k, v in sd.items())
        net = CRAFT()
        net.load_state_dict(strip(torch.load(path, map_location=device)))
        return net.to(device).eval()
    """
    raise NotImplementedError("Wire load_craft() to your CRAFT implementation.")


@torch.no_grad()
def craft_detect_lines(craft_model, image, device):
    """
    >>> TODO <<<
    Run CRAFT on the full prescription `image` (PIL) and return a list of
    cropped line images (PIL) ready for TrOCR. Reuse your existing pre-proc
    + box post-processing (getDetBoxes / adjustResultCoordinates).
    Return: List[PIL.Image]
    """
    raise NotImplementedError("Wire craft_detect_lines() to your CRAFT post-proc.")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trocr", required=True, help="HF TrOCR dir")
    ap.add_argument("--processor", default="microsoft/trocr-small-handwritten",
                    help="processor source if the checkpoint has none "
                         "(base model id, or a local dir for offline use)")
    ap.add_argument("--craft", help="CRAFT .pth (optional; needs TODOs wired)")
    ap.add_argument("--image", help="one full prescription image (for CRAFT + e2e)")
    ap.add_argument("--lines", help="dir of pre-cropped line images (for TrOCR-only)")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=64,
                    help="decoder generation cap; SET THIS TO MATCH the value "
                         "your 0.721 accuracy eval used, so latency & accuracy "
                         "describe the same model behaviour")
    ap.add_argument("--max-lines", type=int, default=50,
                    help="how many distinct test lines to time across")
    ap.add_argument("--inner", type=int, default=3,
                    help="repeats per line (median taken) to suppress jitter")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}\n")

    # ---- TrOCR per-line ----------------------------------------------------
    processor, trocr = load_trocr(args.trocr, device, args.processor)

    if args.lines:
        line_paths = sorted(glob(os.path.join(args.lines, "*")))
        line_imgs = [Image.open(p).convert("RGB") for p in line_paths
                     if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))]
        if not line_imgs:
            ap.error(f"no images found in --lines '{args.lines}' "
                     f"(is the path real? replace the /path/to/... placeholder)")
    elif args.image:
        line_imgs = [Image.open(args.image).convert("RGB")]   # fallback sample
    else:
        ap.error("pass --lines (dir) or --image for the TrOCR timing")

    eval_lines = line_imgs[:args.max_lines]
    print(f"TrOCR (per cropped line, max_new_tokens={args.max_new_tokens})")
    time_across_lines(
        lambda im: trocr_infer_one(processor, trocr, im, device, args.max_new_tokens),
        eval_lines, device, args.warmup, "TrOCR / line", inner=args.inner)

    # ---- CRAFT + end-to-end (only if wired) --------------------------------
    if args.craft and args.image:
        try:
            craft = load_craft(args.craft, device)
            full_img = Image.open(args.image).convert("RGB")

            print("\nCRAFT (per full prescription)")
            time_it(lambda: craft_detect_lines(craft, full_img, device),
                    device, args.runs, args.warmup, "CRAFT / image")

            def end_to_end():
                crops = craft_detect_lines(craft, full_img, device)
                for c in crops:
                    trocr_infer_one(processor, trocr, c, device, args.max_new_tokens)

            print("\nEnd-to-end (CRAFT -> all lines -> TrOCR)")
            time_it(end_to_end, device, max(5, args.runs // 3), args.warmup,
                    "end-to-end / prescription")
        except NotImplementedError as e:
            print(f"\n[skipped CRAFT + end-to-end] {e}")
    else:
        print("\n[skipped CRAFT + end-to-end] pass --craft and --image, and "
              "wire the two CRAFT TODOs.")

    print("\nQuote in thesis: median + p95 as the baseline latency on M4 Pro "
          "(MPS).\nThe same script becomes the on-device number later by "
          "forcing device='cpu' on the Pi.")


if __name__ == "__main__":
    main()