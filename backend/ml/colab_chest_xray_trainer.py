# =========================================================================================
# MEDIEXPLAIN: CHEST X-RAY VISION TRAINER (GOOGLE COLAB VERSION)
# =========================================================================================
# INSTRUCTIONS:
# 1. Open Google Colab (https://colab.research.google.com/)
# 2. Select Runtime -> Change runtime type -> Hardware accelerator -> GPU
# 3. Copy-Paste this entire script into a code cell and run it.
# 4. It will generate 'chest_xray_resnet.pth'. Download it.
# =========================================================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np

# --- 1. SETUP KAGGLE & DOWNLOAD DATA ---
# Replace with your own Kaggle credentials
os.environ['KAGGLE_USERNAME'] = "your_kaggle_username"
os.environ['KAGGLE_KEY'] = "your_kaggle_key"

print("Downloading NIH Chest X-ray Sample Dataset...")
# We use the 'sample' dataset for speed in this demo, full dataset is 40GB+
if not os.path.exists("/content/nih_data"):
    os.system("kaggle datasets download -d nih-chest-xrays/sample -p /content/nih_data --unzip")
else:
    print("Dataset already exists.")

# --- 2. CONFIGURATION ---
DATA_DIR = "/content/nih_data"
# Dynamic Path Finding (Handles Kaggle unzipping variations)
import glob
try:
    img_search = glob.glob(f"{DATA_DIR}/**/images", recursive=True)
    IMG_DIR = img_search[0] if img_search else f"{DATA_DIR}/images"

    csv_search = glob.glob(f"{DATA_DIR}/**/sample_labels.csv", recursive=True)
    CSV_PATH = csv_search[0] if csv_search else f"{DATA_DIR}/sample_labels.csv"

    print(f"✅ Found Image Dir: {IMG_DIR}")
    print(f"✅ Found CSV Path: {CSV_PATH}")
except:
    IMG_DIR = f"{DATA_DIR}/images"
    CSV_PATH = f"{DATA_DIR}/sample_labels.csv"

BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Hardware: {DEVICE}")

# --- 3. DATASET CLASS ---
class ChestXRayDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

        # NIH has 14 diseases, but for demo we focus on 4 common ones
        self.all_labels = ["No Finding", "Pneumonia", "Effusion", "Infiltration"]
        self.label_map = {k: i for i, k in enumerate(self.all_labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image Index']
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')

        # Multi-label classification
        label_vec = torch.zeros(len(self.all_labels))

        findings = row['Finding Labels'].split('|')
        for f in findings:
            if f in self.label_map:
                label_vec[self.label_map[f]] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, label_vec

# --- 4. PREPARE DATA ---
try:
    df = pd.read_csv(CSV_PATH)
    print(f"Total Rows: {len(df)}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = ChestXRayDataset(df, IMG_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 5. MODEL (RESNET50) ---
    print("Initializing ResNet50...")
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    num_ftrs = model.fc.in_features
    # Output layer = 4 classes (No Finding, Pneumonia, Effusion, Infiltration)
    model.fc = nn.Linear(num_ftrs, 4)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()  # Multi-label friendly
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 6. TRAINING LOOP ---
    print("Starting Training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f}")

    # --- 7. SAVE MODEL ---
    save_path = "chest_xray_resnet.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}. Please download this file.")
    
    # --- 8. SAVE TO DIRECTORY FORMAT (Compatible with MediExplain) ---
    # This creates the same structure as your existing models
    os.makedirs("nih_resnet50_multilabel/data", exist_ok=True)
    torch.save(model.state_dict(), "nih_resnet50_multilabel/data.pkl")
    print(f"✅ Also saved in directory format: nih_resnet50_multilabel/")
    print("📦 Download the entire 'nih_resnet50_multilabel' folder and place in backend/ml/models/")

except Exception as e:
    print(f"❌ Error: {e}")
    print("Please ensure you are running this in Google Colab with the dataset downloaded.")
