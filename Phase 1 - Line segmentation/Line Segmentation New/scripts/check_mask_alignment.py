import os
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
# CHECK: Do the old 407 masks still line up with the NEW deskewed images?
#
# Logic: A mask marks WHERE the medicine text is. If the image was
# re-deskewed to a different angle, the text moved, but the mask didn't.
# We measure overlap between each mask's white region and the actual
# dark text pixels in the new ROI image. High overlap = mask still valid.
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT = "/Users/avishkashenan/Desktop/Line Segmentation New"

# The NEW deskewed + cropped images (regenerate ROI from preprocessed_NEW first!)
NEW_ROI_DIR  = os.path.join(PROJECT_ROOT, "data", "roi_images")        # after re-running Step 2
MASK_DIR     = os.path.join(PROJECT_ROOT, "data", "segmentation_masks") # your 407 masks

# How much text must fall inside the mask for it to count as "still aligned"
ALIGNMENT_THRESHOLD = 0.60   # 60% of text pixels inside mask region

masks = [f for f in os.listdir(MASK_DIR)
         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

aligned, misaligned, missing = [], [], []

for mask_name in masks:
    img_path  = os.path.join(NEW_ROI_DIR, mask_name)
    mask_path = os.path.join(MASK_DIR, mask_name)

    if not os.path.exists(img_path):
        missing.append(mask_name)
        continue

    img  = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        missing.append(mask_name)
        continue

    # Match sizes
    if img.shape != mask.shape:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    # Find dark text pixels in the new image
    text = (img < 100).astype(np.uint8)          # dark = text
    mask_bin = (mask > 127).astype(np.uint8)     # white = annotated region

    total_text = text.sum()
    if total_text < 50:
        missing.append(mask_name)
        continue

    # What fraction of the text falls inside the mask?
    text_inside = (text & mask_bin).sum()
    overlap = text_inside / total_text

    if overlap >= ALIGNMENT_THRESHOLD:
        aligned.append(mask_name)
    else:
        misaligned.append((mask_name, round(overlap, 2)))

# ── Report ───────────────────────────────────────────────────
print("=" * 55)
print("MASK ALIGNMENT CHECK")
print("=" * 55)
print(f"  Total masks checked : {len(masks)}")
print(f"  ✅ Still aligned    : {len(aligned)}")
print(f"  ❌ Misaligned       : {len(misaligned)}")
print(f"  ⚠️  Missing/unreadable: {len(missing)}")
print("=" * 55)

pct = len(aligned) / max(1, len(masks)) * 100
print(f"\n  {pct:.0f}% of your annotations still usable.\n")

if pct >= 85:
    print("  VERDICT: Great — keep your masks, just retrain. Re-annotate")
    print("           only the few misaligned ones if you want.")
elif pct >= 60:
    print("  VERDICT: Mostly fine. Keep aligned masks, re-annotate the")
    print("           misaligned batch (listed below).")
else:
    print("  VERDICT: Too many shifted. Safest to re-annotate from the")
    print("           new images (Option 2).")

# Show the worst offenders so you know what to redo
if misaligned:
    print(f"\n  Misaligned files (re-annotate these {len(misaligned)}):")
    for name, ov in sorted(misaligned, key=lambda x: x[1])[:20]:
        print(f"    {name}  (only {int(ov*100)}% text inside)")
    if len(misaligned) > 20:
        print(f"    ... and {len(misaligned) - 20} more")
