import os
import sys
import time
from pathlib import Path
from collections import OrderedDict

# ============================================================================
# HOTFIX FOR MODERN PYTORCH
# CRAFT is from 2019 and looks for an old torchvision variable that was removed.
# We artificially inject it here so the CRAFT code doesn't crash on import!
# ============================================================================
import torchvision.models.vgg as vgg
if not hasattr(vgg, 'model_urls'):
    vgg.model_urls = {'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'}

# ============================================================================
# CRAFT SETUP AND DYNAMIC IMPORTS
# ============================================================================
# Find project root and construct path to the external CRAFT repository
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
craft_repo_dir = os.path.join(project_root, "libs", "CRAFT-pytorch")

# Add the CRAFT repo to Python path so we can import its modules
if craft_repo_dir not in sys.path:
    sys.path.append(craft_repo_dir)

try:
    from craft import CRAFT
    import craft_utils
    import imgproc
except ImportError as e:
    print(f"\n[!] ERROR: Could not import CRAFT modules.")
    print(f"Path searched: {craft_repo_dir}")
    print(f"Exception: {e}")
    print("\nPlease ensure you have cloned the respository correctly:")
    print("  git clone https://github.com/clovaai/CRAFT-pytorch.git libs/CRAFT-pytorch")
    sys.exit(1)

import cv2
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def copyStateDict(state_dict):
    """ Fix for loading weights correctly by removing 'module.' prefix if it exists """
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def load_craft_model(model_path, use_cuda=False):
    """ Loads the pre-trained CRAFT neural network """
    print(f"Loading weights from checkpoint: {model_path}")
    if not os.path.exists(model_path):
         print(f"\n[!] ERROR: Model weights not found at {model_path}")
         print("Please download craft_mlt_25k.pth and place it in the models directory.")
         sys.exit(1)

    net = CRAFT()
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    
    # Mac M-series fallback (MPS) if CUDA isn't available but Apple Silicon is
    if device == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Apple Silicon (MPS) detected! Using hardware acceleration.")
    else:
        print(f"Using device: {device.upper()}")
    
    if device == "cuda":
        net.load_state_dict(copyStateDict(torch.load(model_path)))
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    elif device == "mps":
        net.load_state_dict(copyStateDict(torch.load(model_path, map_location='cpu')))
        net = net.to("mps")
    else:
        net.load_state_dict(copyStateDict(torch.load(model_path, map_location='cpu')))
        
    net.eval()
    return net, device

def get_text_polygons(net, image, params, device):
    """ Runs inference on a single image and extracts word/line polygons """
    # Resize image to fit network constraints while preserving aspect ratio
    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
        image, params['canvas_size'], interpolation=cv2.INTER_LINEAR, mag_ratio=params['mag_ratio']
    )
    ratio_h = ratio_w = 1 / target_ratio
    
    # Preprocessing (Mean Variance Normalization)
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] -> [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] -> [b, c, h, w]
    
    if device == 'cuda':
        x = x.cuda()
    elif device == 'mps':
        x = x.to('mps')

    # Forward Pass
    with torch.no_grad():
        y, _ = net(x)

    # Decode network outputs to score maps
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # Post-processing: Generate bounding boxes from score maps
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, params['text_threshold'], params['link_threshold'], params['low_text'], params['poly']
    )

    # Convert coordinates back to original image scale
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    
    # Ensure polys fallback to rectangular boxes if needed
    for k in range(len(polys)):
        if polys[k] is None: 
            polys[k] = boxes[k]

    return polys

# ============================================================================
# NEW: POST-PROCESSING TO FIX OVER-SEGMENTATION
# ============================================================================
def merge_craft_polygons(polys, merge_threshold=45):
    """
    Takes fragmented CRAFT polygons, sorts them into lines, and mathematically 
    merges boxes that are close to each other on the same horizontal line.
    """
    if len(polys) == 0:
        return []

    # 1. Convert 4-point polygons to standard bounding rects [x, y, w, h]
    rects = []
    for poly in polys:
        pts = np.array(poly, dtype=np.float32)
        x, y, w, h = cv2.boundingRect(pts)
        rects.append([x, y, w, h])

    # 2. Sort top-to-bottom to establish text lines
    rects.sort(key=lambda b: b[1]) 

    lines = []
    current_line = [rects[0]]
    for rect in rects[1:]:
        prev_rect = current_line[-1]
        # If the Y-coordinate is within 50% of the previous box's height, 
        # consider it part of the same handwritten line.
        if abs(rect[1] - prev_rect[1]) < (prev_rect[3] * 0.5): 
            current_line.append(rect)
        else:
            lines.append(current_line)
            current_line = [rect]
    lines.append(current_line)

    # 3. Merge boxes within each line
    merged_rects = []
    for line in lines:
        # Sort left-to-right within the current line
        line.sort(key=lambda b: b[0])

        merged_box = line[0]
        for next_box in line[1:]:
            x1, y1, w1, h1 = merged_box
            x2, y2, w2, h2 = next_box

            # Calculate horizontal gap between the right edge of Box 1 and left edge of Box 2
            gap = x2 - (x1 + w1)

            if gap < merge_threshold:
                # The gap is small! Merge them into a single bounding box.
                new_x = x1
                new_y = min(y1, y2)
                new_w = (x2 + w2) - x1
                new_h = max(y1 + h1, y2 + h2) - new_y
                merged_box = [new_x, new_y, new_w, new_h]
            else:
                # The gap is too large (likely a different column). Keep them separate.
                merged_rects.append(merged_box)
                merged_box = next_box
        merged_rects.append(merged_box)

    # 4. Convert merged rects back to 4-point polygons for your existing crop_and_warp function
    final_polys = []
    for (x, y, w, h) in merged_rects:
        poly = [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ]
        final_polys.append(poly)

    return final_polys

def crop_and_warp(img, poly):
    """
    Crops out a four-point polygon perfectly to a precise rectangular image 
    using a Perspective Transform (great for slightly tilted handwriting).
    """
    pts = np.array(poly, dtype=np.float32)
    
    # Calculate width of new bounding box
    width_top = np.linalg.norm(pts[0] - pts[1])
    width_bottom = np.linalg.norm(pts[3] - pts[2])
    max_width = int(max(width_top, width_bottom))
    
    # Calculate height of new bounding box
    height_left = np.linalg.norm(pts[0] - pts[3])
    height_right = np.linalg.norm(pts[1] - pts[2])
    max_height = int(max(height_left, height_right))
    
    if max_width == 0 or max_height == 0:
        return None
        
    # Desired points for a clean flat rectangle
    dst_pts = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    
    # Execute Perspective Transform
    transform_matrix = cv2.getPerspectiveTransform(pts, dst_pts)
    warped_img = cv2.warpPerspective(img, transform_matrix, (max_width, max_height))
    
    return warped_img

def resize_and_pad(img, target_w=128, target_h=32):
    """
    Resizes image strictly to fit into target_w x target_h while maintaining 
    aspect ratio. Extra space is padded with white. 
    Text is left-aligned and vertically centered.
    """
    if img is None or img.size == 0:
        return None
        
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None
        
    # Scale calculation 
    scale = min(target_w / w, target_h / h)
    
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    # Perform resize
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Initialize white standard canvas
    if len(img.shape) == 3:
        padded = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    else:
        padded = np.ones((target_h, target_w), dtype=np.uint8) * 255
        
    # Calculate positions (left-aligned, vertically centered)
    x_off = 0
    y_off = (target_h - new_h) // 2
    
    # Paste resized crop onto the padded canvas
    if len(img.shape) == 3:
        padded[y_off:y_off+new_h, x_off:x_off+new_w, :] = resized
    else:
        padded[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        
    return padded

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def process_images(input_dir, output_dir, model_path):    
    # Hyperparameters for CRAFT
    params = {
        'text_threshold': 0.7,   # Text confidence threshold
        'link_threshold': 0.4,   # Link confidence threshold
        'low_text': 0.4,         # Text low-bound score
        'cuda': True,            # Try to use CUDA if available
        'poly': False,           # Extract rectangular bounds vs highly complex polys
        'canvas_size': 1280,     # Max image size for inference
        'mag_ratio': 1.5         # Image magnification before inference
    }
    
    net, device = load_craft_model(model_path, use_cuda=params['cuda'])
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    if os.path.exists(input_dir):
        for filename in sorted(os.listdir(input_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(input_dir, filename))
                
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images to process. Output directory: {output_dir}")
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        print(f"Processing {img_name}...")
        
        # We need OpenCV for writing and manipulation, but CRAFT typically takes RGB format
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. Get raw bounding polygons from CRAFT
        raw_polys = get_text_polygons(net, image_rgb, params, device)
        
        # 2. FIX OVER-SEGMENTATION: Merge adjacent words/dosages on the same line
        # Note: If it merges lines that shouldn't be merged, lower the threshold (e.g., to 30).
        # If it's still splitting drug names and dosages, raise the threshold (e.g., to 60).
        polys = merge_craft_polygons(raw_polys, merge_threshold=45)
        
        # Sort polygons geographically: Top-to-Bottom, Left-to-Right 
        # This preserves reading order for handwritten documents
        if len(polys) > 0:
            polys = sorted(polys, key=lambda p: (p[0][1], p[0][0]))
        
        crops_saved = 0
        for i, poly in enumerate(polys):
            # 1. Perspective Crop (compensates for slanting/tilting)
            cropped = crop_and_warp(image, poly)  # Work on original BGR image for saving 
            if cropped is None:
                continue
                
            # 2. Aspect-Ratio Locked Resize & 128x32 Padding
            final_crop = resize_and_pad(cropped, target_w=128, target_h=32)
            if final_crop is None:
                continue
                
            # 3. Save
            base_name = os.path.splitext(img_name)[0]
            out_filename = f"{base_name}_line_{i:03d}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_path, final_crop)
            crops_saved += 1
            
        print(f"  -> Extracted {crops_saved} text regions.")
        
    print("\nSegmentation complete!")

if __name__ == "__main__":
    # Define absolute paths
    INPUT_FOLDER = os.path.join(project_root, "data", "isolated_medicines")
    OUTPUT_FOLDER = os.path.join(project_root, "data", "segmented_lines")
    MODEL_PATH = os.path.join(project_root, "models", "craft_mlt_25k.pth")
    
    process_images(INPUT_FOLDER, OUTPUT_FOLDER, MODEL_PATH)