import cv2
import os
import numpy as np
from pathlib import Path
import csv

def deskew_image(gray):
    """
    Detect skew angle using Hough Transform and rotate image.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    angle = 0.0

    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle_deg = (theta * 180 / np.pi) - 90
            angles.append(angle_deg)

        angle = np.median(angles)

        # Rotate image
        (h, w) = gray.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated, angle

    return gray, angle


def preprocess_images(input_dir, output_dir, log_path):
    """
    Full preprocessing pipeline with logging.
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)

    # Prepare CSV logging
    with open(log_path, mode='w', newline='') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["filename", "status", "skew_angle"])

        image_paths = []

        if os.path.exists(input_dir):
            for filename in os.listdir(input_dir):
                full_path = os.path.join(input_dir, filename)
                if os.path.isfile(full_path) and not filename.startswith('.'):
                    image_paths.append(full_path)

        if not image_paths:
            print(f"No images found in '{input_dir}'")
            return

        print(f"Found {len(image_paths)} files. Starting preprocessing...")

        kernel = np.ones((3, 3), np.uint8)

        for img_path in image_paths:
            filename = os.path.basename(img_path)

            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Skipping unreadable file: {filename}")
                    writer.writerow([filename, "failed_read", "N/A"])
                    continue

                # 1) Grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # 2) Deskew
                gray, angle = deskew_image(gray)

                # 3) Adaptive Thresholding
                thresh = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    15,
                    5
                )

                # 4) Dilation
                dilated = cv2.dilate(thresh, kernel, iterations=1)

                # 5) Inversion
                final_img = cv2.bitwise_not(dilated)

                # Save output
                name_without_ext = os.path.splitext(filename)[0]
                out_path = os.path.join(output_dir, f"{name_without_ext}.jpg")

                cv2.imwrite(out_path, final_img)

                print(f"Processed: {filename} | Angle: {angle:.2f}")

                # Log success
                writer.writerow([filename, "success", round(angle, 2)])

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                writer.writerow([filename, "error", "N/A"])

    print("\n✅ Preprocessing completed!")
    print(f"📁 Images saved to: {output_dir}")
    print(f"📝 Log saved to: {log_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    INPUT_FOLDER = os.path.join(project_root, "data", "raw_images")

    # 🔥 NEW STRUCTURE
    OUTPUT_FOLDER = os.path.join(project_root, "data", "preprocssed 2")
    LOG_FILE = os.path.join(project_root, "data", "processed_output", "logs", "preprocessing_log.csv")

    preprocess_images(INPUT_FOLDER, OUTPUT_FOLDER, LOG_FILE)