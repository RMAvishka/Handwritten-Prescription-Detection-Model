#!/usr/bin/env python3
"""
verify_onnx.py  --  Step 2 check: do the ONNX models match PyTorch?

  * TrOCR : decode the SAME line crops with PyTorch and with the ONNX model;
            report how many produce identical text. Expect (near) 100%.
  * CRAFT : feed the SAME preprocessed image tensor through PyTorch and ONNX;
            report max / mean absolute difference of the score maps.
            Expect tiny diffs (< 1e-3).

If TrOCR text matches and CRAFT diffs are tiny, the export is faithful and we
can move on to timing + INT8. If not, we stop and fix the export.

Run from PROJECT ROOT, `pipeline` env, AFTER export_onnx.py.

USAGE:
    python verify_onnx.py \
        --lines data/cropped_test_images --n 20 \
        --image data/test_images
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import os.path as osp
from glob import glob

import numpy as np
import torch
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def verify_trocr(torch_ckpt, onnx_dir, processor_src, lines_dir, n, max_new_tokens):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from optimum.onnxruntime import ORTModelForVision2Seq

    try:
        proc = TrOCRProcessor.from_pretrained(onnx_dir)
    except Exception:
        proc = TrOCRProcessor.from_pretrained(processor_src)

    pt = VisionEncoderDecoderModel.from_pretrained(torch_ckpt).eval()
    ort = ORTModelForVision2Seq.from_pretrained(onnx_dir)

    paths = sorted(p for p in glob(osp.join(lines_dir, "*"))
                   if p.lower().endswith(IMG_EXT))[:n]
    if not paths:
        print(f"[TrOCR] no line crops in '{lines_dir}' to verify")
        return

    match, mism = 0, []
    for p in paths:
        img = Image.open(p).convert("RGB")
        pv = proc(images=img, return_tensors="pt").pixel_values
        with torch.no_grad():
            a = proc.batch_decode(pt.generate(pv, max_new_tokens=max_new_tokens),
                                  skip_special_tokens=True)[0].strip()
        b = proc.batch_decode(ort.generate(pv, max_new_tokens=max_new_tokens),
                              skip_special_tokens=True)[0].strip()
        if a == b:
            match += 1
        else:
            mism.append((osp.basename(p), a, b))

    print(f"[TrOCR] identical text: {match}/{len(paths)}")
    for f, a, b in mism[:10]:
        print(f"   MISMATCH {f}: torch='{a}' | onnx='{b}'")
    if not mism:
        print("   -> export is faithful for recognition.")


def verify_craft(craft_path, craft_repo, onnx_path, image_path):
    import cv2
    import onnxruntime as ortrt
    from segmentation_pipeline import PrescriptionSegmenter

    seg = PrescriptionSegmenter(craft_model_path=craft_path,
                                craft_repo_dir=craft_repo, device="cpu")
    raw = cv2.imread(image_path)
    if raw is None:
        print(f"[CRAFT] cannot read {image_path}")
        return

    # reproduce the first half of seg._craft_boxes to get the input tensor x
    flat = seg._flatten_paper(raw)
    flat_rgb = cv2.cvtColor(flat, cv2.COLOR_BGR2RGB)
    p = seg.craft_params
    img_resized, _, _ = seg._imgproc.resize_aspect_ratio(
        flat_rgb, p["canvas_size"], interpolation=cv2.INTER_LINEAR,
        mag_ratio=p["mag_ratio"])
    x = seg._imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        y_torch, _ = seg.net(x)
    y_torch = y_torch.cpu().numpy()

    sess = ortrt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    y_onnx = sess.run(None, {"input": x.numpy()})[0]

    if y_torch.shape != y_onnx.shape:
        print(f"[CRAFT] shape MISMATCH: torch {y_torch.shape} vs onnx {y_onnx.shape}")
        return
    diff = np.abs(y_torch - y_onnx)
    print(f"[CRAFT] output shape {y_torch.shape}")
    print(f"[CRAFT] max abs diff {diff.max():.3e} | mean abs diff {diff.mean():.3e}")
    print("   -> faithful if < ~1e-3 (tiny float diffs are expected).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trocr", default="checkpoints/trocr_augmented/best")
    ap.add_argument("--trocr-onnx", default="onnx_models/trocr")
    ap.add_argument("--processor", default="microsoft/trocr-small-handwritten")
    ap.add_argument("--lines", default="data/cropped_test_images")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--craft", default="models/craft_mlt_25k.pth")
    ap.add_argument("--craft-repo", default="libs/CRAFT-pytorch")
    ap.add_argument("--craft-onnx", default="onnx_models/craft/craft.onnx")
    ap.add_argument("--image", default="data/test_images",
                    help="a full prescription image, or a dir (first is used)")
    ap.add_argument("--only", choices=["trocr", "craft"])
    args = ap.parse_args()

    if args.only != "craft":
        verify_trocr(args.trocr, args.trocr_onnx, args.processor,
                     args.lines, args.n, args.max_new_tokens)
    if args.only != "trocr":
        img = args.image
        if osp.isdir(img):
            cands = [p for p in sorted(glob(osp.join(img, "*")))
                     if p.lower().endswith(IMG_EXT)]
            img = cands[0] if cands else None
        if img:
            verify_craft(args.craft, args.craft_repo, args.craft_onnx, img)
        else:
            print("[CRAFT] no image found to verify with")


if __name__ == "__main__":
    main()
