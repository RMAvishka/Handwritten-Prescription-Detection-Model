import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from collections import OrderedDict
from torch.autograd import Variable
from craft import CRAFT
import craft_utils
import imgproc

# ============================================================================
# PATH B: SMART CRAFT + FILTERING  (no DeepLab, no annotation needed)
#
# Idea: CRAFT finds ALL text. We then SCORE each detected region and keep
# only the ones that look like HANDWRITTEN MEDICINE, dropping printed
# clutter (letterhead, patient info, doctor seal, footer).
#
# Features used to tell handwriting from printed text (all training-free):
#   1. Vertical position  - headers/footers live in top & bottom margins
#   2. Stroke regularity  - printed text has uniform stroke width / spacing;
#                           handwriting is irregular
#   3. Ink density        - seals/stamps are very dense round blobs
#   4. Aspect & size      - tiny specks and full-width printed banners filtered
# ============================================================================

PROJECT_ROOT = "/Users/avishkashenan/Desktop/Line Segmentation New"
CRAFT_REPO   = os.path.join(PROJECT_ROOT, "libs", "CRAFT-pytorch")
CRAFT_MODEL  = os.path.join(PROJECT_ROOT, "models", "craft_mlt_25k.pth")

# ─────────────────────────────────────────────────────────────
# FILTER TUNING KNOBS
# ─────────────────────────────────────────────────────────────
TOP_MARGIN_PCT    = 0.12   # drop text in the top 12% of the page (header)
BOTTOM_MARGIN_PCT = 0.12   # drop text in the bottom 12% (footer/signature line)
MIN_REGION_AREA   = 300    # drop tiny specks (px area)
MAX_WIDTH_PCT     = 0.92   # drop near-full-width banners (printed letterhead)
SEAL_DENSITY_MAX  = 0.78   # drop very dense blobs (round seals/stamps)
MERGE_THRESHOLD   = 45     # horizontal gap to merge words on the same line

# ── CRAFT compat fix ─────────────────────────────────────────
import torchvision.models.vgg as vgg
if not hasattr(vgg, 'model_urls'):
    vgg.model_urls = {'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'}

if CRAFT_REPO not in sys.path:
    sys.path.append(CRAFT_REPO)




def copyStateDict(state_dict):
    start = 1 if list(state_dict.keys())[0].startswith("module") else 0
    return OrderedDict({".".join(k.split(".")[start:]): v for k, v in state_dict.items()})


def load_craft():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load(CRAFT_MODEL, map_location='cpu')))
    net = net.to(device)
    net.eval()
    print(f"CRAFT loaded on: {device}")
    return net, device


def get_text_boxes(net, image, device):
    params = {'text_threshold': 0.7, 'link_threshold': 0.4, 'low_text': 0.4,
              'canvas_size': 1280, 'mag_ratio': 1.5, 'poly': False}
    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
        image, params['canvas_size'], interpolation=cv2.INTER_LINEAR, mag_ratio=params['mag_ratio'])
    ratio_h = ratio_w = 1 / target_ratio
    x = imgproc.normalizeMeanVariance(img_resized)
    x = Variable(torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)).to(device)
    with torch.no_grad():
        y, _ = net(x)
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link,
        params['text_threshold'], params['link_threshold'], params['low_text'], params['poly'])
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    return boxes


# ============================================================================
# THE SMART FILTER — this is the "AI reasoning" part of Path B
# ============================================================================
def is_medicine_region(box, gray_img, page_h, page_w):
    """
    Return (keep: bool, reason: str) for one detected text box.
    Scores the region on training-free handwriting-vs-printed features.
    """
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(page_w, x2), min(page_h, y2)

    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1:
        return False, "degenerate"

    # ── Feature 1: vertical position (drop header & footer margins) ──
    cy = (y1 + y2) / 2
    if cy < page_h * TOP_MARGIN_PCT:
        return False, "in_header"
    if cy > page_h * (1 - BOTTOM_MARGIN_PCT):
        return False, "in_footer"

    # ── Feature 2: size sanity ──
    area = w * h
    if area < MIN_REGION_AREA:
        return False, "too_small"
    if w > page_w * MAX_WIDTH_PCT:
        return False, "full_width_banner"   # printed letterhead spans the page

    # ── Feature 3: ink density (seals/stamps are dense round blobs) ──
    roi = gray_img[y1:y2, x1:x2]
    if roi.size == 0:
        return False, "empty"
    ink = (roi < 128).astype(np.uint8)       # dark pixels = ink
    density = ink.sum() / roi.size
    aspect = w / float(h)
    # A near-square, very dense region is almost certainly a seal/stamp
    if density > SEAL_DENSITY_MAX and 0.6 < aspect < 1.7:
        return False, "likely_seal"

    return True, "medicine"


def merge_boxes_into_lines(boxes, threshold=MERGE_THRESHOLD):
    """Same line-merging logic as before: group words on the same row."""
    if len(boxes) == 0:
        return []
    rects = sorted([list(cv2.boundingRect(np.array(b, dtype=np.float32))) for b in boxes],
                   key=lambda b: b[1])
    lines, cur = [], [rects[0]]
    for r in rects[1:]:
        if abs(r[1] - cur[-1][1]) < cur[-1][3] * 0.5:
            cur.append(r)
        else:
            lines.append(cur); cur = [r]
    lines.append(cur)
    merged = []
    for line in lines:
        line.sort(key=lambda b: b[0])
        box = line[0]
        for nb in line[1:]:
            if nb[0] - (box[0] + box[2]) < threshold:
                box = [box[0], min(box[1], nb[1]), (nb[0] + nb[2]) - box[0],
                       max(box[1] + box[3], nb[1] + nb[3]) - min(box[1], nb[1])]
            else:
                merged.append(box); box = nb
        merged.append(box)
    return [[[x, y], [x+w, y], [x+w, y+h], [x, y+h]] for x, y, w, h in merged]


def crop_warp(img, poly):
    pts = np.array(poly, dtype=np.float32)
    mw = int(max(np.linalg.norm(pts[0]-pts[1]), np.linalg.norm(pts[3]-pts[2])))
    mh = int(max(np.linalg.norm(pts[0]-pts[3]), np.linalg.norm(pts[1]-pts[2])))
    if mw == 0 or mh == 0:
        return None
    dst = np.array([[0, 0], [mw-1, 0], [mw-1, mh-1], [0, mh-1]], dtype="float32")
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts, dst), (mw, mh))


def pad_to_size(img, tw=128, th=32):
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None
    s = min(tw/w, th/h)
    nw, nh = max(1, int(w*s)), max(1, int(h*s))
    res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.ones((th, tw, 3) if len(img.shape) == 3 else (th, tw), dtype=np.uint8) * 255
    yo = (th - nh) // 2
    if len(img.shape) == 3:
        out[yo:yo+nh, 0:nw, :] = res
    else:
        out[yo:yo+nh, 0:nw] = res
    return out


def run_path_b(input_dir, output_dir, viz_dir=None):
    """
    Run smart CRAFT+filter on every image.
      input_dir : preprocessed images (NOT DeepLab-isolated — we skip DeepLab!)
      output_dir: final 128x32 line crops
      viz_dir   : optional, saves a visualization showing kept (green) vs
                  dropped (red) regions so you can SEE the filter working
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if viz_dir:
        Path(viz_dir).mkdir(parents=True, exist_ok=True)

    net, device = load_craft()

    images = sorted([f for f in os.listdir(input_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Running Path B on {len(images)} images...\n")

    total_kept, total_dropped = 0, 0

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        page_h, page_w = gray.shape

        # 1. CRAFT detects ALL text
        raw_boxes = get_text_boxes(net, image_rgb, device)

        # 2. SMART FILTER: keep only medicine-looking regions
        kept_boxes = []
        viz = image.copy() if viz_dir else None
        for box in raw_boxes:
            keep, reason = is_medicine_region(box, gray, page_h, page_w)
            if viz is not None:
                pts = np.array(box, dtype=np.int32)
                color = (0, 200, 0) if keep else (0, 0, 255)  # green keep / red drop
                cv2.polylines(viz, [pts], True, color, 2)
            if keep:
                kept_boxes.append(box)
            else:
                total_dropped += 1

        if viz_dir and viz is not None:
            cv2.imwrite(os.path.join(viz_dir, img_name), viz)

        # 3. Merge kept boxes into lines
        lines = merge_boxes_into_lines(kept_boxes)
        lines = sorted(lines, key=lambda p: (p[0][1], p[0][0]))

        # 4. Crop + pad each line to 128x32
        base = os.path.splitext(img_name)[0]
        saved = 0
        for i, poly in enumerate(lines):
            cropped = crop_warp(image, poly)
            final = pad_to_size(cropped, 128, 32)
            if final is None:
                continue
            cv2.imwrite(os.path.join(output_dir, f"{base}_line_{i:03d}.jpg"), final)
            saved += 1
            total_kept += 1

        print(f"  {img_name}: kept {saved} medicine lines "
              f"(dropped {len(raw_boxes) - len(kept_boxes)} clutter regions)")

    print(f"\nDONE. {total_kept} medicine line crops saved to: {output_dir}")
    print(f"Total clutter regions filtered out: {total_dropped}")
    if viz_dir:
        print(f"\nVisualizations (green=kept, red=dropped) saved to: {viz_dir}")
        print("Open a few of these to SEE the filter deciding. Tune knobs at top if needed.")


if __name__ == "__main__":
    # NOTE: we run on preprocessed_images directly — DeepLab is SKIPPED.
    INPUT  = os.path.join(PROJECT_ROOT, "data", "preprocessed_images")
    OUTPUT = os.path.join(PROJECT_ROOT, "data", "segmented_lines_pathB")
    VIZ    = os.path.join(PROJECT_ROOT, "data", "pathB_visualization")

    run_path_b(INPUT, OUTPUT, viz_dir=VIZ)
