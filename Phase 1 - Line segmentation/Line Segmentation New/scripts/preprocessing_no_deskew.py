import cv2
import os
import numpy as np
from pathlib import Path

# ============================================================================
# PREPROCESSING (NO DESKEW VERSION)
#
# Paper-deskew already flattened & uprighted the page, so we do NOT deskew
# again here. This step only does: grayscale -> threshold -> dilate -> invert.
#
# INPUT : data/paper_flattened/   (output of paper_deskew.py)
# OUTPUT: data/preprocessed_images/
# ============================================================================

PROJECT_ROOT = "/Users/avishkashenan/Desktop/Line Segmentation New"


def preprocess_images(input_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_paths = []
    if os.path.exists(input_dir):
        for filename in os.listdir(input_dir):
            full_path = os.path.join(input_dir, filename)
            if os.path.isfile(full_path) and not filename.startswith('.'):
                image_paths.append(full_path)

    if not image_paths:
        print(f"No images found in '{input_dir}'")
        return

    print(f"Found {len(image_paths)} files. Preprocessing (no deskew)...")
    kernel = np.ones((3, 3), np.uint8)

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Skipping unreadable: {filename}")
            continue

        # 1) Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2) Adaptive threshold (NO deskew — paper is already flat)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 5
        )

        # 3) Dilate
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # 4) Invert (back to black text on white)
        final_img = cv2.bitwise_not(dilated)

        name = os.path.splitext(filename)[0]
        out_path = os.path.join(output_dir, f"{name}.jpg")
        cv2.imwrite(out_path, final_img)
        print(f"  Processed: {filename}")

    print(f"\nDone! Saved to: {output_dir}")


if __name__ == "__main__":
    INPUT  = os.path.join(PROJECT_ROOT, "data", "paper_flattened")
    OUTPUT = os.path.join(PROJECT_ROOT, "data", "preprocessed_images")
    preprocess_images(INPUT, OUTPUT)
