import cv2
import os
import numpy as np
from pathlib import Path

# ============================================================================
# PAPER-DETECTION DESKEW
#
# Instead of guessing skew from text (which fails on clutter), we detect the
# PRESCRIPTION PAPER itself — the bright sheet sitting on a darker background —
# find its 4 corners, and warp it flat with a perspective transform.
#
# This fixes BOTH small tilts and heavy rotations (the 30-40 deg cases),
# because we map the paper's actual corners to a clean upright rectangle.
#
# Run this on RAW images FIRST, then feed its output into your normal
# preprocessing (grayscale/threshold) and Path B.
# ============================================================================

PROJECT_ROOT = "/Users/avishkashenan/Desktop/Line Segmentation New"


def order_corners(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left  = smallest x+y
    rect[2] = pts[np.argmax(s)]   # bot-right = largest  x+y
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # top-right = smallest y-x
    rect[3] = pts[np.argmax(d)]   # bot-left  = largest  y-x
    return rect


def find_paper_corners(img, min_area_ratio=0.20):
    """
    Find the 4 corners of the paper sheet.
    Returns ordered corners, or None if no good quadrilateral found.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur then threshold: paper is bright, background is dark
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # Otsu separates bright paper from dark surroundings
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Close gaps so the paper becomes one solid blob
    kernel = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Largest contour = the paper (hopefully)
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # Reject if the "paper" is too small to be real
    if area < (h * w * min_area_ratio):
        return None

    # Approximate the contour to a polygon
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) == 4:
        # Clean 4-corner paper found
        return order_corners(approx.reshape(4, 2))
    else:
        # Not a clean quad — fall back to minAreaRect (rotated bounding box)
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        return order_corners(box)


def warp_paper(img, corners):
    """Warp the detected paper to a flat upright rectangle."""
    (tl, tr, br, bl) = corners

    # Compute output size from the corner distances
    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    if maxW < 50 or maxH < 50:
        return None

    dst = np.array([
        [0, 0], [maxW - 1, 0],
        [maxW - 1, maxH - 1], [0, maxH - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH))
    return warped


def deskew_by_paper(input_dir, output_dir, debug_dir=None):
    """
    Detect + flatten paper for every raw image.
    Saves flattened images. If paper can't be found, saves the original
    unchanged (so nothing is lost).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)

    images = sorted([f for f in os.listdir(input_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Paper-deskew on {len(images)} images...\n")

    found, fallback = 0, 0

    for img_name in images:
        img = cv2.imread(os.path.join(input_dir, img_name))
        if img is None:
            continue

        corners = find_paper_corners(img)

        if corners is not None:
            warped = warp_paper(img, corners)
            if warped is not None:
                cv2.imwrite(os.path.join(output_dir, img_name), warped)
                found += 1

                # Debug: draw detected corners on original
                if debug_dir:
                    dbg = img.copy()
                    cv2.polylines(dbg, [corners.astype(np.int32)], True, (0, 255, 0), 4)
                    for p in corners:
                        cv2.circle(dbg, tuple(p.astype(int)), 12, (0, 0, 255), -1)
                    cv2.imwrite(os.path.join(debug_dir, img_name), dbg)
                continue

        # Fallback: no paper found, keep original so we don't lose the image
        cv2.imwrite(os.path.join(output_dir, img_name), img)
        fallback += 1
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, img_name), img)

    print(f"DONE.")
    print(f"  Paper detected + flattened : {found}")
    print(f"  No paper found (kept as-is): {fallback}")
    print(f"  Output: {output_dir}")
    if debug_dir:
        print(f"\n  Debug images (green outline + red corners): {debug_dir}")
        print(f"  Open these to verify the paper corners were found correctly.")
        print(f"  If corners are wrong on many images, we tune the detection.")


if __name__ == "__main__":
    # Run on RAW images — this is the very first step now, BEFORE preprocessing
    RAW_INPUT   = os.path.join(PROJECT_ROOT, "data", "raw_images")
    FLATTENED   = os.path.join(PROJECT_ROOT, "data", "paper_flattened")
    DEBUG       = os.path.join(PROJECT_ROOT, "data", "paper_debug")

    deskew_by_paper(RAW_INPUT, FLATTENED, debug_dir=DEBUG)
