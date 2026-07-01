# import os
# import cv2
# import torch
# import numpy as np
# from torchvision import transforms
# from torchvision.models.segmentation import deeplabv3_resnet50
# from pathlib import Path
# from tqdm import tqdm



# def build_model(model_path, device, num_classes=2):
#     """ Loads the DeepLabV3+ architecture and injects your trained weights. """
#     print("Loading your trained DeepLab model...")
#     import torch.nn as nn
    
#     # THE FIX: Add aux_loss=True so the architecture perfectly matches your saved weights
#     model = deeplabv3_resnet50(weights=None, aux_loss=True)
    
#     # Modify the heads to match our 2 classes
#     model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
#     if model.aux_classifier is not None:
#         model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
#     # Inject your trained brain!
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval() # Set to evaluation mode (turns off training features)
#     return model

# def run_inference():
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.dirname(script_dir)
    
#     # We will test it on the images you already have
#     INPUT_DIR = os.path.join(project_root, "data", "preprocessed_images")
#     OUTPUT_DIR = os.path.join(project_root, "data", "isolated_medicines_new")
#     MODEL_PATH = os.path.join(project_root, "models", "deeplab_medicine_detector.pth")
    
#     Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

#     # Hardware detection
#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#     elif torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")

#     model = build_model(MODEL_PATH, device)

#     # The exact same image transformation used during training
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     print(f"\nRunning inference on {len(images)} images...")

#     for img_name in tqdm(images):
#         img_path = os.path.join(INPUT_DIR, img_name)
        
#         # 1. Load Original Image
#         original_img = cv2.imread(img_path)
#         if original_img is None:
#             continue
            
#         orig_h, orig_w = original_img.shape[:2]

#         # 2. Prepare for DeepLab (Must be 512x512 RGB)
#         img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
#         img_resized = cv2.resize(img_rgb, (512, 512))
#         input_tensor = transform(img_resized).unsqueeze(0).to(device) # Add batch dimension

#         # 3. Model Prediction!
#         with torch.no_grad():
#             output = model(input_tensor)['out'][0]
#             # Get the predicted class for each pixel (0 for background, 1 for medicine)
#             predicted_mask = output.argmax(0).byte().cpu().numpy() 

#         # 4. Scale mask back up to the original high-resolution image size
#         # We use INTER_NEAREST to keep the edges sharp (no blurry grays)
#         full_res_mask = cv2.resize(predicted_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

#         # 5. Apply the mask! (Keep original pixels where mask is 1, otherwise make black)
#         isolated_img = cv2.bitwise_and(original_img, original_img, mask=full_res_mask)

#         # 6. Save the perfectly isolated medicine region
#         out_path = os.path.join(OUTPUT_DIR, img_name)
#         cv2.imwrite(out_path, isolated_img)

# if __name__ == "__main__":
#     run_inference()





import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from pathlib import Path
from tqdm import tqdm


def clean_mask(mask):
    """
    Post-process DeepLab's raw mask so you get ONE clean medicine region
    instead of scattered green patches.

    Steps:
      1. Morphological close — fills small holes inside the region.
      2. Keep only the largest connected component — removes stray blobs.
      3. Light dilation — expands slightly so no text gets clipped at edges.
    """
    if mask.sum() == 0:
        return mask  # nothing detected, return as-is

    # 1) Close small gaps/holes
    kernel = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 2) Keep only the largest blob
    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num > 1:
        # stats[0] is the background; find the largest non-background component
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cleaned = (labels == largest).astype(np.uint8)
    else:
        cleaned = closed

    # 3) Slight dilation so we don't clip edge text
    cleaned = cv2.dilate(cleaned, np.ones((9, 9), np.uint8), iterations=1)

    return cleaned


def build_model(model_path, device, num_classes=2):
    """Loads the DeepLabV3+ architecture and injects trained weights."""
    print("Loading trained DeepLab model...")
    model = deeplabv3_resnet50(weights=None, aux_loss=True)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def run_inference():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    INPUT_DIR = os.path.join(project_root, "data", "preprocessed_images")
    OUTPUT_DIR = os.path.join(project_root, "data", "isolated_medicines_new")
    MODEL_PATH = os.path.join(project_root, "models", "deeplab_medicine_detector.pth")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = build_model(MODEL_PATH, device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\nRunning inference on {len(images)} images...")

    for img_name in tqdm(images):
        img_path = os.path.join(INPUT_DIR, img_name)
        original_img = cv2.imread(img_path)
        if original_img is None:
            continue

        orig_h, orig_w = original_img.shape[:2]
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 512))
        input_tensor = transform(img_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)['out'][0]
            predicted_mask = output.argmax(0).byte().cpu().numpy()

        # Scale mask up to original resolution
        full_res_mask = cv2.resize(predicted_mask, (orig_w, orig_h),
                                   interpolation=cv2.INTER_NEAREST)

        # *** THE NEW FIX: clean the mask ***
        full_res_mask = clean_mask(full_res_mask)

        isolated_img = cv2.bitwise_and(original_img, original_img, mask=full_res_mask)

        out_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(out_path, isolated_img)

    print("Inference complete with cleaned masks!")


if __name__ == "__main__":
    run_inference()