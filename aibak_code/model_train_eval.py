import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from dataset_preperation import RoadSkeletonDataset
from sklearn.metrics import jaccard_score
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Modified U-Net Architecture
class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (Downsampling)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder (Upsampling) with corrected channel dimensions
        self.up1 = Up(1024 + 512, 512, bilinear)  # Fixed input channels
        self.up2 = Up(512 + 256, 256, bilinear)
        self.up3 = Up(256 + 128, 128, bilinear)
        self.up4 = Up(128 + 64, 64, bilinear)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # 64 channels
        x2 = self.down1(x1)   # 128
        x3 = self.down2(x2)   # 256
        x4 = self.down3(x3)   # 512
        x5 = self.down4(x4)   # 1024

        # Decoder with proper skip connections
        x = self.up1(x5, x4)  # 1024+512=1536 -> 512
        x = self.up2(x, x3)   # 512+256=768 -> 256
        x = self.up3(x, x2)   # 256+128=384 -> 128
        x = self.up4(x, x1)   # 128+64=192 -> 64
        
        return self.outc(x)

# 2. Training Configuration
class Config:
    def __init__(self):
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.epochs = 100
        self.input_size = (256, 256)
        self.threshold = 0.5
        self.model_save_path = "unet_skeleton.pth"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# 3. Training Loop
def train_model(train_loader, val_loader, model, criterion, optimizer, config):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            inputs = batch['input'].float().to(config.device)
            targets = batch['target'].float().to(config.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].float().to(config.device)
                targets = batch['target'].float().to(config.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.model_save_path)
    
    return train_losses, val_losses

# 4. Evaluation Metrics
def calculate_metrics(preds, targets, threshold=0.5):
    preds_bin = (preds > threshold).float()
    targets = targets.float()
    
    tp = (preds_bin * targets).sum()
    fp = (preds_bin * (1 - targets)).sum()
    fn = ((1 - preds_bin) * targets).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    
    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'dice': dice.item()
    }

# 5. Evaluation Script
def evaluate_model(test_loader, model, config):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].float().to(config.device)
            targets = batch['target'].cpu().numpy()
            
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy()
            
            all_preds.append(preds)
            all_targets.append(targets)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Flatten arrays for sklearn metrics
    preds_flat = all_preds.reshape(-1) > config.threshold
    targets_flat = all_targets.reshape(-1) > config.threshold
    
    return {
        'iou': jaccard_score(targets_flat, preds_flat),
        'precision': precision_score(targets_flat, preds_flat),
        'recall': recall_score(targets_flat, preds_flat),
        'f1': f1_score(targets_flat, preds_flat)
    }

# 6. Main Execution
if __name__ == "__main__":
    # Initialize dataset and loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = RoadSkeletonDataset(
        thinning_dir='/home/aljadaaa/Documents/skelnton/Road_Skeletonization/thinning_data/data/thinning',
        target_png_dir='/home/aljadaaa/Documents/skelnton/Road_Skeletonization/thinning_data/data/targets_png',
        geojson_dir = '/home/aljadaaa/Documents/skelnton/Road_Skeletonization/thinning_data/data/targets_geojson',
        transform=transform
    )
    
    val_dataset = RoadSkeletonDataset(
        thinning_dir='/home/aljadaaa/Documents/skelnton/Road_Skeletonization/thinning_data/data/thinning',
        target_png_dir='/home/aljadaaa/Documents/skelnton/Road_Skeletonization/thinning_data/data/targets_png',
        geojson_dir = '/home/aljadaaa/Documents/skelnton/Road_Skeletonization/thinning_data/data/targets_geojson',
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Initialize model
    model = UNet(n_channels=1, n_classes=1).to(config.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Train
    train_losses, val_losses = train_model(train_loader, val_loader, model, criterion, optimizer, config)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Progress')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Evaluate
    test_dataset = RoadSkeletonDataset(
        thinning_dir='path/to/test_thinning',
        target_png_dir='path/to/test_targets',
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    metrics = evaluate_model(test_loader, model, config)
    print("\nTest Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Save predictions visual
    sample = test_dataset[0]
    with torch.no_grad():
        pred = torch.sigmoid(model(sample['input'].unsqueeze(0).to(config.device))).cpu().numpy()
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(sample['input'].squeeze(), cmap='gray')
    plt.title('Input')
    plt.subplot(1, 3, 2)
    plt.imshow(pred.squeeze() > config.threshold, cmap='gray')
    plt.title('Prediction')
    plt.subplot(1, 3, 3)
    plt.imshow(sample['target'].squeeze(), cmap='gray')
    plt.title('Ground Truth')
    plt.savefig('sample_prediction.png')
    plt.show()
