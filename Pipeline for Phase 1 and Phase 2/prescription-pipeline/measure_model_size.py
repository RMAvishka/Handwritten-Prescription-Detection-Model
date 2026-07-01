#!/usr/bin/env python3
"""
measure_model_size.py
=====================
Baseline (a): on-disk size + parameter count for the Phase-4 models.

Reports, per model:
  - parameter count
  - actual size on disk (the checkpoint files you ship)
  - THEORETICAL weight memory at FP32 / FP16 / INT8
    (params x bytes-per-param; this is the *weights only* lower bound,
     not runtime RAM, which also includes activations + framework overhead)

Works for:
  - TrOCR  : a HuggingFace directory (config.json + *.safetensors/*.bin)
             OR a single .pth/.bin state_dict
  - CRAFT  : a single .pth state_dict (e.g. craft_mlt_25k.pth)

Usage
-----
  python measure_model_size.py \
      --trocr /path/to/your/finetuned-trocr-dir \
      --craft /path/to/craft_mlt_25k.pth

You can pass either, both, or point --trocr at a single .pth file.
No GPU needed; this only inspects weights.
"""

import argparse
import os
from collections import OrderedDict

import torch

BYTES_PER_PARAM = {"fp32": 4, "fp16": 2, "int8": 1}


def _iter_tensors(obj):
    """Yield every torch.Tensor inside a state_dict / nested dict / list."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_tensors(v)


def count_params_from_state_dict(path):
    """Load a .pth/.bin checkpoint and count total parameters."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Common wrappers: {'model': sd}, {'state_dict': sd}, or raw sd
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "net"):
            if key in ckpt and isinstance(ckpt[key], (dict, OrderedDict)):
                ckpt = ckpt[key]
                break
    return sum(t.numel() for t in _iter_tensors(ckpt))


def count_params_from_hf(path):
    """Load a HuggingFace TrOCR directory and count total parameters."""
    from transformers import VisionEncoderDecoderModel

    model = VisionEncoderDecoderModel.from_pretrained(path)
    total = sum(p.numel() for p in model.parameters())
    # also break encoder/decoder so you can quote the split in the thesis
    enc = sum(p.numel() for p in model.encoder.parameters())
    dec = sum(p.numel() for p in model.decoder.parameters())
    return total, enc, dec


def disk_size_bytes(path):
    """Size on disk: a single file, or the sum of weight files in a dir."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    weight_ext = (".safetensors", ".bin", ".pth", ".pt")
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(weight_ext):
                total += os.path.getsize(os.path.join(root, f))
    return total


def mb(n_bytes):
    return n_bytes / (1024 ** 2)


def theoretical_table(n_params):
    return {dt: n_params * b / (1024 ** 2) for dt, b in BYTES_PER_PARAM.items()}


def report(name, path, is_hf):
    print("=" * 60)
    print(f"  {name}")
    print(f"  path: {path}")
    print("=" * 60)

    enc = dec = None
    if is_hf and os.path.isdir(path):
        n_params, enc, dec = count_params_from_hf(path)
    else:
        n_params = count_params_from_state_dict(path)

    disk = disk_size_bytes(path)
    theo = theoretical_table(n_params)

    print(f"  parameters         : {n_params:,} ({n_params/1e6:.2f} M)")
    if enc is not None:
        print(f"    - encoder        : {enc:,} ({enc/1e6:.2f} M)")
        print(f"    - decoder        : {dec:,} ({dec/1e6:.2f} M)")
    print(f"  on-disk size       : {mb(disk):.2f} MB")
    print(f"  theoretical weights:")
    print(f"    - FP32           : {theo['fp32']:.2f} MB")
    print(f"    - FP16           : {theo['fp16']:.2f} MB")
    print(f"    - INT8           : {theo['int8']:.2f} MB"
          f"   (~{theo['fp32']/theo['int8']:.1f}x smaller than FP32)")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trocr", help="HF TrOCR dir OR a .pth state_dict")
    ap.add_argument("--craft", help="CRAFT .pth state_dict")
    args = ap.parse_args()

    if not args.trocr and not args.craft:
        ap.error("pass --trocr and/or --craft")

    if args.trocr:
        report("TrOCR (Phase 2)", args.trocr, is_hf=True)
    if args.craft:
        report("CRAFT (Phase 1)", args.craft, is_hf=False)

    print("Note: 'theoretical weights' is the lower bound for the weights "
          "alone.\nActual peak RAM on the Pi is higher (activations, the "
          "decoder's\nautoregressive cache, Python + runtime overhead). "
          "Measure that\nseparately on-device in the deployment step.")


if __name__ == "__main__":
    main()
