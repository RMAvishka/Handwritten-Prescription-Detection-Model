#!/usr/bin/env python3
"""
prune_finetune_trocr.py -- Step 4: structured pruning of TrOCR + fine-tune.

Removes whole DECODER LAYERS (the only pruning that speeds up CPU inference),
keeping evenly-spaced layers (DistilBERT-style), then fine-tunes briefly on your
train.csv so the survivors re-adapt. Saves a pruned+fine-tuned checkpoint that
then goes back through export_onnx.py -> quantize_trocr.py -> eval_trocr_accuracy.py
for the same size/speed/accuracy measurement as the un-pruned model.

WHY decoder layers: TrOCR's decoder runs once PER GENERATED CHARACTER, so fewer
layers = proportionally less work every step = real latency win (unlike scattered
weight pruning, which doesn't speed up dense CPU math).

Run from PROJECT ROOT, `pipeline` env.

FIRST validate it runs on a tiny subset:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python prune_finetune_trocr.py \
        --drop 2 --limit 50 --epochs 1

Then the real run (all training rows):
    PYTORCH_ENABLE_MPS_FALLBACK=1 python prune_finetune_trocr.py \
        --drop 2 --epochs 3
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def find_decoder(model):
    """Return the TrOCRDecoder module that holds the .layers ModuleList."""
    # VisionEncoderDecoderModel -> decoder (TrOCRForCausalLM) -> model -> decoder
    td = model.decoder.model.decoder
    assert hasattr(td, "layers"), "could not find decoder.layers"
    return td


def prune_decoder_layers(model, drop):
    td = find_decoder(model)
    old = len(td.layers)
    keep = max(1, old - drop)
    if keep >= old:
        return old, old
    # evenly spaced indices, always including the first and last layer
    if keep == 1:
        idx = [0]
    else:
        idx = sorted({round(i * (old - 1) / (keep - 1)) for i in range(keep)})
    td.layers = nn.ModuleList([td.layers[i] for i in idx])
    # keep configs in sync so generate() uses the new depth
    for cfg in (getattr(model, "config", None),
                getattr(model.decoder, "config", None)):
        if cfg is not None and hasattr(cfg, "decoder_layers"):
            cfg.decoder_layers = len(idx)
        if cfg is not None and hasattr(cfg, "decoder") and hasattr(cfg.decoder, "decoder_layers"):
            cfg.decoder.decoder_layers = len(idx)
    print(f"  decoder layers: {old} -> {len(idx)}  (kept indices {idx})")
    return old, len(idx)


class LineDS(Dataset):
    def __init__(self, df, root, processor, img_col, label_col, max_len):
        self.rows = []
        for _, r in df.iterrows():
            p = root / "images" / str(r[img_col])
            if p.exists():
                self.rows.append((p, str(r[label_col]).strip().lower()))
        self.processor = processor
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, text = self.rows[i]
        img = Image.open(p).convert("RGB")
        pv = self.processor(images=img, return_tensors="pt").pixel_values[0]
        tok = self.processor.tokenizer(
            text, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt")
        labels = tok.input_ids[0]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return pv, labels


def collate(batch):
    pvs = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return pvs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/trocr_augmented/best")
    ap.add_argument("--processor", default="microsoft/trocr-small-handwritten")
    ap.add_argument("--train-csv", default="data/pharmacy_lk/splits/train.csv")
    ap.add_argument("--data-root", default="data/pharmacy_lk")
    ap.add_argument("--img-col", default="image_filename")
    ap.add_argument("--label-col", default="medicine_name")
    ap.add_argument("--out", default="checkpoints/trocr_pruned/best")
    ap.add_argument("--drop", type=int, default=2, help="decoder layers to remove")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows (quick test: 50)")
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    try:
        processor = TrOCRProcessor.from_pretrained(args.ckpt)
    except Exception:
        processor = TrOCRProcessor.from_pretrained(args.processor)
    model = VisionEncoderDecoderModel.from_pretrained(args.ckpt)

    n_before = sum(p.numel() for p in model.parameters())
    prune_decoder_layers(model, args.drop)
    n_after = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_before/1e6:.2f}M -> {n_after/1e6:.2f}M "
          f"({(1 - n_after/n_before)*100:.1f}% smaller)")

    # set special tokens for generation/loss
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.to(device)

    df = pd.read_csv(args.train_csv).dropna(subset=[args.label_col])
    if args.limit:
        df = df.iloc[:args.limit]
    ds = LineDS(df, Path(args.data_root), processor, args.img_col,
               args.label_col, args.max_len)
    print(f"  training samples: {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x  # graceful no-op if tqdm isn't installed

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for ep in range(args.epochs):
        running, steps = 0.0, 0
        bar = tqdm(loader, desc=f"epoch {ep+1}/{args.epochs}", unit="batch")
        for pv, labels in bar:
            pv, labels = pv.to(device), labels.to(device)
            loss = model(pixel_values=pv, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            running += loss.item()
            steps += 1
            # live loss on the bar; tqdm shows elapsed, ETA, and it/s
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(loss=f"{running/steps:.4f}")
        print(f"  epoch {ep+1} done | mean loss {running/max(steps,1):.4f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"\nsaved pruned+fine-tuned model -> {args.out}")
    print("Next: re-export -> re-quantize -> re-eval this checkpoint, e.g.")
    print(f"  python export_onnx.py --only trocr --trocr {args.out} "
          f"--trocr-out onnx_models/trocr_pruned")
    print("  (then quantize_trocr.py + eval_trocr_accuracy.py pointed at the "
          "pruned dirs)")


if __name__ == "__main__":
    main()
