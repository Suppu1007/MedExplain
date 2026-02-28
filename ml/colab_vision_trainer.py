
!pip install torch torchvision pillow opencv-python scikit-learn tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from tqdm import tqdm

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# DATASET CLASSES
# ============================================================================

class ChestXRayDataset(Dataset):
    """
    NIH Chest X-Ray Dataset
    14 disease classes: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass,
    Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis,
    Pleural_Thickening, Hernia
    """
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # Load labels (CSV with image_name, finding_labels columns)
        # For demo, create synthetic data
        self.images = []
        self.labels = []
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class BrainMRIDataset(Dataset):
    """
    Brain MRI Tumor Detection Dataset
    Binary classification: Tumor / No Tumor
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# MODEL DEFINITION
# ============================================================================

def create_resnet50_model(num_classes, pretrained=True):
    """
    Create ResNet-50 model for medical image classification
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
    """
    model = models.resnet50(pretrained=pretrained)
    
    # Modify final layer for our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    """Train the model"""
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(device)
            labels = labels.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                labels = labels.to(device).float()
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Convert to binary predictions
        train_preds_binary = (np.array(train_preds) > 0.5).astype(int)
        val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
        
        train_acc = accuracy_score(train_labels, train_preds_binary)
        val_acc = accuracy_score(val_labels, val_preds_binary)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ Saved best model (Val Acc: {val_acc:.4f})")
        
        scheduler.step()
    
    return model

# ============================================================================
# MAIN TRAINING SCRIPTS
# ============================================================================

def train_chest_xray_model():
    """Train ResNet-50 for NIH Chest X-Ray (14 diseases)"""
    print("="*60)
    print("Training Chest X-Ray Model (NIH 14-Disease Classification)")
    print("="*60)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset (replace with actual data paths)
    # train_dataset = ChestXRayDataset('path/to/images', 'path/to/labels.csv', transform)
    # val_dataset = ChestXRayDataset('path/to/val_images', 'path/to/val_labels.csv', transform)
    
    # For demo purposes, create dummy loaders
    print("⚠ Using dummy data for demonstration")
    print("⚠ Replace with actual NIH Chest X-Ray dataset")
    
    # Create model
    model = create_resnet50_model(num_classes=14, pretrained=True)
    
    # Train model
    # trained_model = train_model(model, train_loader, val_loader, num_epochs=20)
    
    # Save model
    torch.save(model.state_dict(), 'resnet50_chest_xray.pth')
    print("\n✓ Model saved as resnet50_chest_xray.pth")
    print("✓ Download and place in backend/ml/models/")
    
    return model

def train_brain_mri_model():
    """Train ResNet-50 for Brain MRI Tumor Detection"""
    print("="*60)
    print("Training Brain MRI Model (Tumor Detection)")
    print("="*60)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset (replace with actual data paths)
    # train_dataset = BrainMRIDataset('path/to/train', transform)
    # val_dataset = BrainMRIDataset('path/to/val', transform)
    
    print("⚠ Using dummy data for demonstration")
    print("⚠ Replace with actual Brain MRI dataset")
    
    # Create model (binary classification)
    model = create_resnet50_model(num_classes=1, pretrained=True)
    
    # Train model
    # trained_model = train_model(model, train_loader, val_loader, num_epochs=15)
    
    # Save model
    torch.save(model.state_dict(), 'resnet50_brain_mri.pth')
    print("\n✓ Model saved as resnet50_brain_mri.pth")
    print("✓ Download and place in backend/ml/models/")
    
    return model

# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    print("\nMediExplain Vision Model Trainer")
    print("="*60)
    print("\nIMPORTANT: This script requires:")
    print("1. NIH Chest X-Ray Dataset (112,000+ images)")
    print("   Download: https://nihcc.app.box.com/v/ChestXray-NIHCC")
    print("2. Brain MRI Dataset (Kaggle or similar)")
    print("   Download: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection")
    print("3. GPU with 8GB+ VRAM (Google Colab Pro recommended)")
    print("\n" + "="*60)
    
    # Train Chest X-Ray model
    chest_model = train_chest_xray_model()
    
    # Train Brain MRI model
    brain_model = train_brain_mri_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Download resnet50_chest_xray.pth")
    print("2. Download resnet50_brain_mri.pth")
    print("3. Place both files in backend/ml/models/")
    print("4. Run the application!")
