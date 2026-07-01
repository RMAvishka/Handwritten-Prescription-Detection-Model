import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from collections import OrderedDict
from torch.autograd import Variable

# ============================================================================
# PATH B v2: SMART CRAFT + HANDWRITING DISCRIMINATOR + LINE MERGING
#
# Two upgrades over v1:
#   FIX 1 (printed-text filter): drop the doctor's printed credential block
#         wherever it sits, using stroke-regularity features (no training).
#   FIX 2 (line merging): stitch word-boxes on the same handwritten line into
#         ONE crop (e.g. "Apixiban 2.5 mg bd x 2/12" => single line).
# ============================================================================

PROJECT_ROOT = "/Users/avishkashenan/Desktop/Line Segmentation New"
CRAFT_REPO   = os.path.join(PROJECT_ROOT, "libs", "CRAFT-pytorch")
CRAFT_MODEL  = os.path.join(PROJECT_ROOT, "models", "craft_mlt_25k.pth")

# ─────────────────────────────────────────────────────────────
# FILTER TUNING KNOBS
# ─────────────────────────────────────────────────────────────
TOP_MARGIN_PCT    = 0.10
BOTTOM_MARGIN_PCT = 0.08
MIN_REGION_AREA   = 300
MAX_WIDTH_PCT     = 0.92
SEAL_DENSITY_MAX  = 0.78

# Printed-text discriminator knobs (FIX 1)
PRINTED_FILTER_ON     = True
PRINTED_ROW_STD_MAX   = 0.16   # printed text has very regular row-ink profile
                               #   (low variation). Below this = looks printed.
PRINTED_MIN_HEIGHT    = 12     # only judge "printed" on reasonably tall regions
                               #   (avoids killing tiny handwriting)

# Line merging knobs (FIX 2)
SAME_LINE_Y_TOL   = 0.7   # two boxes are on the same line if their vertical
                          #   centers differ by < this * average height
LINE_MERGE_GAP    = 120   # horizontal gap (px) allowed between words on a line
                          #   BIG because handwritten dose/freq are far apart.
                          #   v1 used 45 — that's why lines were getting split.

# ── CRAFT compat fix ─────────────────────────────────────────
import torchvision.models.vgg as vgg
if not hasattr(vgg, 'model_urls'):
    vgg.model_urls = {'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'}

if CRAFT_REPO not in sys.path:
    sys.path.append(CRAFT_REPO)

from craft import CRAFT
import craft_utils
import imgproc


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
# FIX 1: PRINTED vs HANDWRITTEN DISCRIMINATOR
# ============================================================================
def looks_printed(roi_gray):
    """
    Heuristic: printed text has a very REGULAR horizontal ink profile —
    every row of the text band has similar ink amount, because the font
    sits on a clean baseline with uniform x-height. Handwriting is irregular:
    ascenders, descenders, and uneven pressure make the row profile noisy.

    We measure the coefficient of variation of the per-row ink count over
    the ink-bearing rows. LOW variation => printed.
    """
    h, w = roi_gray.shape
    if h < PRINTED_MIN_HEIGHT or w < 10:
        return False  # too small to judge — treat as handwriting (keep)

    ink = (roi_gray < 128).astype(np.float32)
    row_ink = ink.sum(axis=1)                 # ink per row
    active = row_ink[row_ink > 0]             # ignore blank rows
    if len(active) < 4:
        return False

    mean = active.mean()
    if mean < 1e-6:
        return False
    cov = active.std() / mean                 # coefficient of variation

    # Low COV = very uniform rows = printed text
    return cov < PRINTED_ROW_STD_MAX


def box_to_rect(box, page_h, page_w):
    xs = [p[0] for p in box]; ys = [p[1] for p in box]
    x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(page_w, x2), min(page_h, y2)
    return x1, y1, x2, y2


def is_medicine_region(box, gray_img, page_h, page_w):
    """Return (keep, reason) for one detected box."""
    x1, y1, x2, y2 = box_to_rect(box, page_h, page_w)
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1:
        return False, "degenerate"

    cy = (y1 + y2) / 2
    if cy < page_h * TOP_MARGIN_PCT:
        return False, "in_header"
    if cy > page_h * (1 - BOTTOM_MARGIN_PCT):
        return False, "in_footer"

    area = w * h
    if area < MIN_REGION_AREA:
        return False, "too_small"
    if w > page_w * MAX_WIDTH_PCT:
        return False, "full_width_banner"

    roi = gray_img[y1:y2, x1:x2]
    if roi.size == 0:
        return False, "empty"

    ink = (roi < 128).astype(np.uint8)
    density = ink.sum() / roi.size
    aspect = w / float(h)
    if density > SEAL_DENSITY_MAX and 0.6 < aspect < 1.7:
        return False, "likely_seal"

    # FIX 1: drop printed credential blocks wherever they are
    if PRINTED_FILTER_ON and looks_printed(roi):
        return False, "printed_text"

    return True, "medicine"


# ============================================================================
# FIX 2: BETTER LINE MERGING
# ============================================================================
def merge_into_lines(boxes, page_h, page_w):
    """Group kept word-boxes into full handwritten lines."""
    if len(boxes) == 0:
        return []

    rects = []
    for b in boxes:
        x1, y1, x2, y2 = box_to_rect(b, page_h, page_w)
        rects.append([x1, y1, x2 - x1, y2 - y1])

    # sort top-to-bottom
    rects.sort(key=lambda r: r[1])

    # group into rows by vertical center proximity
    lines, cur = [], [rects[0]]
    for r in rects[1:]:
        prev = cur[-1]
        avg_h = (prev[3] + r[3]) / 2.0
        prev_cy = prev[1] + prev[3] / 2
        r_cy    = r[1] + r[3] / 2
        if abs(r_cy - prev_cy) < avg_h * SAME_LINE_Y_TOL:
            cur.append(r)
        else:
            lines.append(cur); cur = [r]
    lines.append(cur)

    # within each row, merge left-to-right when gap is small enough
    merged = []
    for line in lines:
        line.sort(key=lambda r: r[0])
        box = line[0]
        for nb in line[1:]:
            gap = nb[0] - (box[0] + box[2])
            if gap < LINE_MERGE_GAP:
                nx = box[0]
                ny = min(box[1], nb[1])
                nx2 = max(box[0] + box[2], nb[0] + nb[2])
                ny2 = max(box[1] + box[3], nb[1] + nb[3])
                box = [nx, ny, nx2 - nx, ny2 - ny]
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
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if viz_dir:
        Path(viz_dir).mkdir(parents=True, exist_ok=True)

    net, device = load_craft()

    images = sorted([f for f in os.listdir(input_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Running Path B v2 on {len(images)} images...\n")

    total_lines = 0
    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        page_h, page_w = gray.shape

        raw_boxes = get_text_boxes(net, image_rgb, device)

        kept = []
        viz = image.copy() if viz_dir else None
        for box in raw_boxes:
            keep, reason = is_medicine_region(box, gray, page_h, page_w)
            if viz is not None:
                pts = np.array(box, dtype=np.int32)
                color = (0, 200, 0) if keep else (0, 0, 255)
                cv2.polylines(viz, [pts], True, color, 2)
            if keep:
                kept.append(box)

        # merge kept words into lines
        lines = merge_into_lines(kept, page_h, page_w)
        lines = sorted(lines, key=lambda p: (p[0][1], p[0][0]))

        # draw final merged lines in BLUE on the viz so you can see merging
        if viz is not None:
            for poly in lines:
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(viz, [pts], True, (255, 150, 0), 3)
            cv2.imwrite(os.path.join(viz_dir, img_name), viz)

        base = os.path.splitext(img_name)[0]
        saved = 0
        for i, poly in enumerate(lines):
            cropped = crop_warp(image, poly)
            final = pad_to_size(cropped, 128, 32)
            if final is None:
                continue
            cv2.imwrite(os.path.join(output_dir, f"{base}_line_{i:03d}.jpg"), final)
            saved += 1
            total_lines += 1

        print(f"  {img_name}: {saved} medicine lines "
              f"({len(raw_boxes)} raw boxes -> {len(kept)} kept -> {saved} merged lines)")

    print(f"\nDONE. {total_lines} medicine line crops saved to: {output_dir}")
    if viz_dir:
        print(f"\nViz saved to: {viz_dir}")
        print("  GREEN  = kept word   RED = dropped   BLUE = final merged line")
        print("  Check that BLUE boxes wrap whole medicine lines, and the")
        print("  printed doctor credentials are now RED.")


if __name__ == "__main__":
    INPUT  = os.path.join(PROJECT_ROOT, "data", "preprocessed_images")
    OUTPUT = os.path.join(PROJECT_ROOT, "data", "segmented_lines_pathB_v2")
    VIZ    = os.path.join(PROJECT_ROOT, "data", "pathB_v2_visualization")
    run_path_b(INPUT, OUTPUT, viz_dir=VIZ)
