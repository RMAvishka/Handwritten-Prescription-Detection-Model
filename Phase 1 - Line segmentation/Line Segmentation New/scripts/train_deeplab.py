import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from tqdm import tqdm 
import numpy as np
from pathlib import Path

# ============================================================================
# 1. CUSTOM DATASET LOADER
# ============================================================================
class PrescriptionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(512, 512)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        
        # NEW LOGIC: Cross-reference folders. Only keep images that have a matching mask!
        raw_images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.images = []
        
        for img_name in raw_images:
            mask_path = os.path.join(self.mask_dir, img_name)
            if os.path.exists(mask_path):
                self.images.append(img_name)
                
        print(f"Dataset Loader: Found {len(self.images)} valid annotated pairs (Skipped {len(raw_images) - len(self.images)} images).")
        
        # Data Augmentation for the Image
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Load Image (RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)

        # Load Mask (Grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        # Convert mask to binary (0 for Background, 1 for Medicine)
        mask = (mask > 127).astype(np.int64)

        # Apply transforms
        image = self.img_transform(image)
        mask = torch.from_numpy(mask)

        return image, mask

# ============================================================================
# 2. MODEL BUILDER
# ============================================================================
def build_model(num_classes=2):
    """ Loads pre-trained DeepLabV3+ and modifies the output for our 2 classes. """
    print("Downloading/Loading pre-trained DeepLabV3+ (ResNet50) backbone...")
    
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
    return model

# ============================================================================
# 3. TRAINING ENGINE
# ============================================================================
def train_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    IMG_DIR = os.path.join(project_root, "data", "roi_images")
    MASK_DIR = os.path.join(project_root, "data", "segmentation_masks")
    MODEL_SAVE_DIR = os.path.join(project_root, "models")
    Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    
    # Hyperparameters
    BATCH_SIZE = 4       
    EPOCHS = 15          
    LEARNING_RATE = 1e-4 
    
    # 1. Hardware Detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple Silicon (M4 Pro) MPS detected! Engaging hardware acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("⚠️ Warning: No GPU detected. Training will be slow.")

    # 2. Prepare Data
    print("Preparing dataset...")
    dataset = PrescriptionDataset(IMG_DIR, MASK_DIR)
    
    if len(dataset) == 0:
        print("Error: No valid image-mask pairs found!")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Initialize Model, Loss, and Optimizer
    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"\nStarting training for {EPOCHS} epochs on {len(dataset)} valid images...")
    print("-" * 50)

    # 4. The Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs['out'], masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f}\n")

    # 5. Save the trained brain
    save_path = os.path.join(MODEL_SAVE_DIR, "deeplab_medicine_detector.pth")
    torch.save(model.state_dict(), save_path)
    print("Training Complete! Model saved successfully to:")
    print(f"   {save_path}")

if __name__ == "__main__":
    train_model()