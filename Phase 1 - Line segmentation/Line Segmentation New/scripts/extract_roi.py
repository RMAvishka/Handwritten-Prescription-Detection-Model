import cv2
import os
import numpy as np
from pathlib import Path

# ============================================================================
# ROI EXTRACTION STRATEGIES
# ============================================================================

def crop_by_percentage(image, top_pct=0.25, bottom_pct=0.20):
    """
    Assumes a standard layout and blindly slices off the top and bottom.
    Extremely fast and reliable if the documents are relatively uniform.
    """
    h, w = image.shape[:2]
    
    # Calculate pixel rows to slice
    start_y = int(h * top_pct)
    end_y = int(h * (1.0 - bottom_pct))
    
    # Crop and return
    return image[start_y:end_y, 0:w]


def crop_by_rx_template(image, template_path, bottom_pct=0.20, threshold=0.7):
    """
    BONUS: Uses OpenCV Template Matching to search for an 'Rx' symbol.
    Crops everything below the symbol and above the footer percentage.
    If the 'Rx' isn't found with high enough confidence, it falls back to percentage cropping.
    """
    h, w = image.shape[:2]
    end_y = int(h * (1.0 - bottom_pct))
    
    # Load template
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print("    [!] Warning: Rx Template not found. Falling back to percentage crop.")
        return crop_by_percentage(image, 0.25, bottom_pct)
        
    th, tw = template.shape[:2]
    
    # Convert image to grayscale for matching
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Perform matching
    result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if max_val >= threshold:
        # The bottom of the Rx symbol becomes our new top crop line
        rx_bottom_y = max_loc[1] + th
        
        # Add a tiny 20px padding below the symbol just to be safe
        start_y = min(rx_bottom_y + 20, end_y - 100) 
        return image[start_y:end_y, 0:w]
    else:
        print(f"    [!] Rx symbol confidence ({max_val:.2f}) below threshold. Falling back to percentage.")
        return crop_by_percentage(image, 0.25, bottom_pct)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def extract_roi_from_folder(input_dir, output_dir, strategy="percentage"):
    """ Reads images, applies the chosen ROI strategy, and saves them. """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_paths = []
    if os.path.exists(input_dir):
        for filename in sorted(os.listdir(input_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(input_dir, filename))
                
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Starting ROI Extraction ({strategy} strategy) on {len(image_paths)} images...")
    
    crops_saved = 0
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"  -> Skipping {filename} (Could not read)")
            continue
            
        # Apply the selected strategy
        if strategy == "rx_template":
            # NOTE: You must crop an 'Rx' symbol yourself and save it as 'rx_template.jpg' in your models folder
            template_path = os.path.join(project_root, "models", "rx_template.jpg")
            roi_image = crop_by_rx_template(image, template_path)
        else:
            # Default to fixed percentages (Top 25%, Bottom 20%)
            roi_image = crop_by_percentage(image, top_pct=0.25, bottom_pct=0.20)
            
        # Save the resulting ROI
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, roi_image)
        crops_saved += 1
        
    print(f"ROI Extraction complete! Saved {crops_saved} cropped images to {output_dir}")

if __name__ == "__main__":
    # Dynamically find project paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    INPUT_FOLDER = os.path.join(project_root, "data", "preprocessed_images")
    OUTPUT_FOLDER = os.path.join(project_root, "data", "roi_images")
    
    # Run the extraction. 
    # Change strategy to "rx_template" if you want to try the bonus method!
    extract_roi_from_folder(INPUT_FOLDER, OUTPUT_FOLDER, strategy="percentage")