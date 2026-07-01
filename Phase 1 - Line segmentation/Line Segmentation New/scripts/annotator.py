import cv2
import os
import numpy as np
from pathlib import Path

# --- Global Variables ---
points = []          
img_display = None   
img_original = None  

def draw_polygon(event, x, y, flags, param):
    """ Mouse callback to let the user click 4 corners to make a slanted box. """
    global points, img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            
            if len(points) > 1:
                cv2.line(img_display, points[-2], points[-1], (0, 255, 0), 2)
            
            if len(points) == 4:
                cv2.line(img_display, points[3], points[0], (0, 255, 0), 2)
                overlay = img_display.copy()
                cv2.fillPoly(overlay, [np.array(points)], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, img_display, 0.7, 0, img_display)

            cv2.imshow('Annotator', img_display)

def draw_hud(img, current, total, annotated, skipped):
    """ Draws the Heads Up Display with dataset stats and instructions. """
    text_color = (0, 0, 0)
    bg_color = (255, 255, 255)
    
    # Text lines to display
    lines = [
        f"PROGRESS: Image {current} of {total}",
        f"Annotated: {annotated} | Skipped: {skipped}",
        "-" * 30,
        "Click 4 corners around medicine.",
        "[S] Save  |  [N] Skip Image",
        "[C] Clear |  [Q] Save & Quit"
    ]
    
    # Draw white background box
    cv2.rectangle(img, (10, 10), (360, 180), bg_color, -1)
    cv2.rectangle(img, (10, 10), (360, 180), (0, 0, 0), 2) # Black border
    
    y = 35
    for i, line in enumerate(lines):
        # Make the progress stats slightly bolder
        thickness = 2 if i < 2 else 1
        cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, thickness)
        y += 25

def run_annotator(input_dir, mask_dir, skipped_log_path):
    global img_display, img_original, points
    
    Path(mask_dir).mkdir(parents=True, exist_ok=True)
    
    # Load skipped images memory
    skipped_set = set()
    if os.path.exists(skipped_log_path):
        with open(skipped_log_path, 'r') as f:
            skipped_set = set(f.read().splitlines())
    
    # Gather all images
    all_images = []
    if os.path.exists(input_dir):
        for filename in sorted(os.listdir(input_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(filename)
                
    total_images = len(all_images)
    if total_images == 0:
        print("No images found to annotate.")
        return

    # Count how many are already annotated
    annotated_count = len([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    skipped_count = len(skipped_set)

    cv2.namedWindow('Annotator', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Annotator', 800, 1000) 
    cv2.setMouseCallback('Annotator', draw_polygon)

    for index, filename in enumerate(all_images, start=1):
        img_path = os.path.join(input_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        
        # Resume Logic: Skip if already annotated or marked as skipped previously
        if os.path.exists(mask_path) or filename in skipped_set:
            continue

        img_original = cv2.imread(img_path)
        if img_original is None:
            continue
            
        points = [] 
        img_display = img_original.copy()
        draw_hud(img_display, index, total_images, annotated_count, skipped_count)

        while True:
            cv2.imshow('Annotator', img_display)
            key = cv2.waitKey(1) & 0xFF

            # [S] Save
            if key == ord('s'):
                if len(points) >= 3: 
                    h, w = img_original.shape[:2]
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [np.array(points)], 255)
                    cv2.imwrite(mask_path, mask)
                    
                    annotated_count += 1
                    print(f"[SAVED] {filename}")
                    break
                else:
                    print("Please click 4 corners first!")
                    
            # [C] Clear
            elif key == ord('c'):
                points = []
                img_display = img_original.copy()
                draw_hud(img_display, index, total_images, annotated_count, skipped_count)
                print("Cleared. Try again.")
                
            # [N] Skip
            elif key == ord('n'):
                skipped_set.add(filename)
                skipped_count += 1
                # Save to the log file immediately
                with open(skipped_log_path, 'a') as f:
                    f.write(filename + '\n')
                print(f"[SKIPPED] {filename}")
                break
                
            # [Q] Quit
            elif key == ord('q'):
                print(f"\nSession saved! You have annotated {annotated_count} images total.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("\nDataset Complete! You have finished all images.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    INPUT_FOLDER = os.path.join(project_root, "data", "preprocessed_images") 
    MASK_FOLDER = os.path.join(project_root, "data", "segmentation_masks")
    SKIPPED_LOG = os.path.join(project_root, "data", "skipped_images.txt")
    
    run_annotator(INPUT_FOLDER, MASK_FOLDER, SKIPPED_LOG)