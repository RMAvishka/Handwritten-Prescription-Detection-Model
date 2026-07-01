import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# TUNING KNOBS — adjust these two if the mask is still cutting text
# ─────────────────────────────────────────────────────────────
MEDICINE_THRESHOLD = 0.30   # Lower = bigger mask (catches more text).
                            #   0.50 = old argmax behavior
                            #   0.30 = include pixels model is 30%+ sure about
                            #   0.20 = even more generous
DILATE_PIXELS      = 25     # How many px to expand the final region outward.
                            #   Bigger = safer margin around text. 9 was old value.
KEEP_LARGEST_ONLY  = True   # Keep only the biggest blob (kills stray dots).
                            #   Set False if medicine text is split across
                            #   two separate areas of the page.


def clean_mask(mask, dilate_px=DILATE_PIXELS, keep_largest=KEEP_LARGEST_ONLY):
    """Clean the mask but bias toward KEEPING text (expand, don't shrink)."""
    if mask.sum() == 0:
        return mask

    # 1) Close holes
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    # 2) Optionally keep only the largest blob
    if keep_largest:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        if num > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            closed = (labels == largest).astype(np.uint8)

    # 3) Strong dilation — expand outward so we don't clip edge text
    cleaned = cv2.dilate(closed, np.ones((dilate_px, dilate_px), np.uint8), iterations=1)

    return cleaned


def build_model(model_path, device, num_classes=2):
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
    OUTPUT_DIR = os.path.join(project_root, "data", "isolated_medicines_v2")
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
    print(f"  Threshold: {MEDICINE_THRESHOLD}  |  Dilation: {DILATE_PIXELS}px  |  Largest-only: {KEEP_LARGEST_ONLY}")

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
            output = model(input_tensor)['out'][0]          # shape: [2, 512, 512]
            # *** KEY CHANGE: use softmax probability, not argmax ***
            probs = F.softmax(output, dim=0)                  # convert to probabilities
            medicine_prob = probs[1].cpu().numpy()            # prob of "medicine" class
            predicted_mask = (medicine_prob >= MEDICINE_THRESHOLD).astype(np.uint8)

        # Scale up to original resolution
        full_res_mask = cv2.resize(predicted_mask, (orig_w, orig_h),
                                   interpolation=cv2.INTER_NEAREST)

        # Clean + expand
        full_res_mask = clean_mask(full_res_mask)

        isolated_img = cv2.bitwise_and(original_img, original_img, mask=full_res_mask)

        out_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(out_path, isolated_img)

    print(f"\nDone! Saved to: {OUTPUT_DIR}")
    print("Compare against your old folder. If text is STILL cut off,")
    print(f"lower MEDICINE_THRESHOLD (try 0.20) or raise DILATE_PIXELS (try 35).")


if __name__ == "__main__":
    run_inference()
