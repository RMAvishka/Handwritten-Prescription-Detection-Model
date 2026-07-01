# import cv2
# import os
# import numpy as np
# from pathlib import Path

# def preprocess_images(input_dir, output_dir):
#     """
#     Reads handwritten medical prescription images from input_dir, applies
#     preprocessing steps (grayscale, adaptive thresholding, and morphological
#     dilation), and saves them to output_dir.
#     """
#     # Create the output directory if it doesn't already exist
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
    
#     # NEW FIX: Grab every file in the directory, regardless of extension
#     image_paths = []
#     if os.path.exists(input_dir):
#         for filename in os.listdir(input_dir):
#             full_path = os.path.join(input_dir, filename)
#             # Make sure it's a file, not a hidden sub-folder (like .DS_Store on Mac)
#             if os.path.isfile(full_path) and not filename.startswith('.'):
#                 image_paths.append(full_path)
    
#     if not image_paths:
#         print(f"No images found in '{input_dir}'. Please ensure the folder exists and contains images.")
#         return

#     print(f"Found {len(image_paths)} files. Starting preprocessing...")

#     # Define a 3x3 kernel for morphological operations
#     kernel = np.ones((3, 3), np.uint8)

#     for img_path in image_paths:
#         # Read the image
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"Warning: Could not read {img_path}. Skipping.")
#             continue
            
#         # 1) Convert to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#         # 2) Apply adaptive thresholding to binarize the image and handle uneven lighting.
#         thresh = cv2.adaptiveThreshold(
#             gray, 
#             255, 
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#             cv2.THRESH_BINARY_INV, 
#             15, # Block size (neighborhood area size, should be an odd number)
#             5   # Constant subtracted from the calculated mean 
#         )
        
#         # 3) Apply a morphological dilation operation using the 3x3 kernel.
#         dilated = cv2.dilate(thresh, kernel, iterations=1)
        
#         # 4) Invert the image back so the background is white and the text is black.
#         final_img = cv2.bitwise_not(dilated)
        
#         # Extract filename and construct the output save path
#         filename = os.path.basename(img_path)
        
#         # Optional: Force the output to be saved as a .jpg to standardize your dataset
#         name_without_ext = os.path.splitext(filename)[0]
#         out_path = os.path.join(output_dir, f"{name_without_ext}.jpg")
        
#         # Save the resulting cleaned image into the output folder
#         cv2.imwrite(out_path, final_img)
#         print(f"Processed: {filename}")

#     print(f"\nPreprocessing completed! Cleaned images saved to '{output_dir}'.")

# if __name__ == "__main__":
#     # 1. Get absolute paths
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.dirname(script_dir)
    
#     # 2. Construct paths to data folders
#     INPUT_FOLDER = os.path.join(project_root, "data", "raw_images")
#     OUTPUT_FOLDER = os.path.join(project_root, "data", "preprocessed_images")
    
#     # 3. Run the preprocessing function
#     preprocess_images(INPUT_FOLDER, OUTPUT_FOLDER)

















import cv2
import os
import numpy as np
from pathlib import Path
import csv


def deskew_image(gray, max_angle=15.0):
    """
    Robust deskew using the orientation of the TEXT itself (not background clutter).

    How it works:
      1. Threshold to find dark text pixels on light paper.
      2. Find the minimum-area rotated rectangle around all text pixels.
      3. That rectangle's angle = the text's tilt.
      4. Clamp to +/- max_angle. If the detected angle is larger than that,
         we assume detection failed and DO NOT rotate (return angle 0).

    This avoids the old Hough-lines problem where table edges, paper borders,
    and the dark background produced random 45-90 degree "skew" angles.
    """
    # Binarize: text becomes white (255) on black background
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Coordinates of all text pixels
    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) < 50:
        # Almost no text found — don't risk rotating
        return gray, 0.0

    # minAreaRect returns angle in range [-90, 0)
    angle = cv2.minAreaRect(coords)[-1]

    # Normalize the angle to a small correction value
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # SAFETY CLAMP: real prescriptions are only slightly tilted.
    # If we detected something extreme, detection failed — skip rotation.
    if abs(angle) > max_angle:
        return gray, 0.0

    # Rotate by the (small, safe) detected angle
    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle


def preprocess_images(input_dir, output_dir, log_path, max_angle=15.0):
    """Full preprocessing pipeline with robust deskew + logging."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)

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

        skipped_rotations = 0

        for img_path in image_paths:
            filename = os.path.basename(img_path)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"  Skipping unreadable file: {filename}")
                    writer.writerow([filename, "failed_read", "N/A"])
                    continue

                # 1) Grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # 2) Robust Deskew
                gray, angle = deskew_image(gray, max_angle=max_angle)
                if angle == 0.0:
                    skipped_rotations += 1

                # 3) Adaptive Thresholding
                thresh = cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 15, 5
                )

                # 4) Dilation
                dilated = cv2.dilate(thresh, kernel, iterations=1)

                # 5) Inversion
                final_img = cv2.bitwise_not(dilated)

                # Save
                name_without_ext = os.path.splitext(filename)[0]
                out_path = os.path.join(output_dir, f"{name_without_ext}.jpg")
                cv2.imwrite(out_path, final_img)

                print(f"  Processed: {filename} | Angle: {angle:.2f}")
                writer.writerow([filename, "success", round(angle, 2)])

            except Exception as e:
                print(f"  Error processing {filename}: {str(e)}")
                writer.writerow([filename, "error", "N/A"])

    print(f"\n Preprocessing completed!")
    print(f"  Images saved to: {output_dir}")
    print(f"  Log saved to:    {log_path}")
    print(f"  Images left un-rotated (angle too extreme / no text): {skipped_rotations}")
    print(f"  -> A high number here is GOOD if your photos were already straight.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    INPUT_FOLDER = os.path.join(project_root, "data", "raw_images")
    OUTPUT_FOLDER = os.path.join(project_root, "data", "preprocessed_images")
    LOG_FILE = os.path.join(project_root, "data", "logs", "preprocessing_log.csv")

    preprocess_images(INPUT_FOLDER, OUTPUT_FOLDER, LOG_FILE, max_angle=15.0)