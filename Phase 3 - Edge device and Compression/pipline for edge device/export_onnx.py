#!/usr/bin/env python3
"""
export_onnx.py  --  Step 2 (NO compression yet): export both models to ONNX.

  * TrOCR : exported with HuggingFace Optimum (ORTModelForVision2Seq), which
            handles the encoder / decoder / decoder-with-cache graphs and keeps
            .generate() working. The processor is saved alongside so the ONNX
            folder is SELF-CONTAINED (important for the offline Pi).
  * CRAFT : exported with torch.onnx.export via your PrescriptionSegmenter
            (reuses your exact weights + the vgg compat fix). Dynamic H/W so it
            accepts the variable canvas sizes your pipeline produces.

Run from the PROJECT ROOT, in the `pipeline` env, AFTER installing:
    pip install "optimum[onnxruntime]"

USAGE:
    python export_onnx.py                 # export both
    python export_onnx.py --only trocr    # just TrOCR
    python export_onnx.py --only craft     # just CRAFT
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import os.path as osp

import torch


def size_mb(path):
    if osp.isfile(path):
        return osp.getsize(path) / 1024 / 1024
    tot = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            tot += osp.getsize(osp.join(r, f))
    return tot / 1024 / 1024


def export_trocr(ckpt, processor_src, out_dir):
    from optimum.onnxruntime import ORTModelForVision2Seq
    from transformers import TrOCRProcessor

    print(f"[TrOCR] exporting {ckpt} -> {out_dir}  (a few minutes; downloads "
          f"nothing if cached)")
    model = ORTModelForVision2Seq.from_pretrained(ckpt, export=True)
    model.save_pretrained(out_dir)

    # self-contained: save the processor into the ONNX dir
    try:
        proc = TrOCRProcessor.from_pretrained(ckpt)
        print("[TrOCR] processor: from checkpoint")
    except Exception:
        proc = TrOCRProcessor.from_pretrained(processor_src)
        print(f"[TrOCR] processor: from '{processor_src}' (checkpoint had none)")
    proc.save_pretrained(out_dir)

    print(f"[TrOCR] done. ONNX dir total: {size_mb(out_dir):.2f} MB")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".onnx"):
            print(f"    - {f}: {size_mb(osp.join(out_dir, f)):.2f} MB")


def export_craft(craft_path, craft_repo, out_path, height, width, opset):
    import onnx
    from segmentation_pipeline import PrescriptionSegmenter

    print(f"[CRAFT] loading via PrescriptionSegmenter (cpu) ...")
    seg = PrescriptionSegmenter(craft_model_path=craft_path,
                                craft_repo_dir=craft_repo, device="cpu")
    net = seg.net.cpu().eval()

    out_dir = osp.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    dummy = torch.randn(1, 3, height, width, dtype=torch.float32)
    print(f"[CRAFT] exporting -> {out_path}  (dummy {height}x{width}, opset {opset})")
    torch.onnx.export(
        net, dummy, out_path,
        input_names=["input"], output_names=["score", "feature"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "score": {0: "batch", 1: "h_out", 2: "w_out"}},
        opset_version=opset, do_constant_folding=True)

    # The new (dynamo) exporter may store weights as EXTERNAL data, leaving a
    # tiny .onnx + sidecar files. Re-load (resolves the external data from
    # out_dir) and re-save everything INTERNAL -> one self-contained file.
    model = onnx.load(out_path)
    onnx.save_model(model, out_path, save_as_external_data=False)

    # delete the now-orphaned external-data sidecars (weights are embedded now)
    for f in os.listdir(out_dir):
        fp = osp.join(out_dir, f)
        if osp.isfile(fp) and fp != out_path and not f.endswith(".onnx"):
            os.remove(fp)

    print(f"[CRAFT] done. self-contained ONNX size: {size_mb(out_path):.2f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trocr", default="checkpoints/trocr_augmented/best")
    ap.add_argument("--processor", default="microsoft/trocr-small-handwritten")
    ap.add_argument("--trocr-out", default="onnx_models/trocr")
    ap.add_argument("--craft", default="models/craft_mlt_25k.pth")
    ap.add_argument("--craft-repo", default="libs/CRAFT-pytorch")
    ap.add_argument("--craft-out", default="onnx_models/craft/craft.onnx")
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--only", choices=["trocr", "craft"], help="export one model")
    args = ap.parse_args()

    if args.only != "craft":
        export_trocr(args.trocr, args.processor, args.trocr_out)
    if args.only != "trocr":
        export_craft(args.craft, args.craft_repo, args.craft_out,
                     args.height, args.width, args.opset)
    print("\nNext: python verify_onnx.py   (confirms ONNX == PyTorch outputs)")


if __name__ == "__main__":
    main()