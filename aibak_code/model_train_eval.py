import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from dataset_preperation import RoadSkeletonDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt, convolve
from scipy.spatial.distance import cdist
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
"""

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
"""
def compute_mse_distance_transform(pred, target):
    pred_bin = (pred > 0.5).astype(np.uint8)
    target_bin = (target > 0.5).astype(np.uint8)
    print(pred_bin)
    dist_transform = distance_transform_edt(1 - target_bin)
    mse = np.mean((pred_bin * dist_transform) ** 2)
    return mse

def compute_valence_map(skeleton):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    return convolve(skeleton.astype(np.uint8), kernel, mode='constant')

def extract_nodes_by_valence(skeleton, valence):
    valence_map = compute_valence_map(skeleton)
    return np.argwhere((skeleton > 0) & (valence_map == valence))

def match_nodes(pred_nodes, gt_nodes, tolerance=3):
    if len(pred_nodes) == 0 or len(gt_nodes) == 0:
        return 0, 0, 0
    dists = cdist(pred_nodes, gt_nodes)
    matched_pred = set()
    matched_gt = set()
    for i, row in enumerate(dists):
        for j, dist in enumerate(row):
            if dist <= tolerance and j not in matched_gt and i not in matched_pred:
                matched_pred.add(i)
                matched_gt.add(j)
                break
    tp = len(matched_pred)
    fp = len(pred_nodes) - tp
    fn = len(gt_nodes) - tp
    return tp, fp, fn

def evaluate_node_precision_recall(pred_skeleton, gt_skeleton):
    results = {}
    for v in [1, 2, 3, 4]:
        pred_nodes = extract_nodes_by_valence(pred_skeleton, v)
        gt_nodes = extract_nodes_by_valence(gt_skeleton, v)
        tp, fp, fn = match_nodes(pred_nodes, gt_nodes)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        results[f'valence_{v}_precision'] = precision
        results[f'valence_{v}_recall'] = recall
    return results

def compute_iou_and_dice(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).astype(np.uint8)
    target_bin = (target > threshold).astype(np.uint8)
    intersection = np.logical_and(pred_bin, target_bin).sum()
    union = np.logical_or(pred_bin, target_bin).sum()
    dice = 2. * intersection / (pred_bin.sum() + target_bin.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    return iou, dice

def evaluate_model(test_loader, model, config):
    model.eval()
    total_loss = 0
    total_mse = 0
    total_iou = 0
    total_dice = 0
    valence_metrics_sum = {f'valence_{v}_precision': 0 for v in range(1, 5)}
    valence_metrics_sum.update({f'valence_{v}_recall': 0 for v in range(1, 5)})
    count = 0

    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].float().to(config.device)
            targets = batch['target'].float().to(config.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            preds = torch.sigmoid(outputs).cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            for i in range(inputs.shape[0]):

                pred = preds[i, 0]
                target = targets_np[i, 0]
                mse = compute_mse_distance_transform(pred, target)
                valence_metrics = evaluate_node_precision_recall(pred > config.threshold, target > config.threshold)
                iou, dice = compute_iou_and_dice(pred, target, threshold=config.threshold)

                total_mse += mse
                total_iou += iou
                total_dice += dice
                for k in valence_metrics:
                    valence_metrics_sum[k] += valence_metrics[k]

                count += 1

    metrics = {
        'test_loss': total_loss / len(test_loader),
        'mse': total_mse / count,
        'iou': total_iou / count,
        'dice': total_dice / count,
    }
    for k in valence_metrics_sum:
        metrics[k] = valence_metrics_sum[k] / count

    return metrics

# 6. Main Execution
if __name__ == "__main__":
    # Initialize dataset and loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = RoadSkeletonDataset(
        thinning_dir='/home/aljadaaa/Documents/skelnton/Road_Skeletonization/thinning_data/data/thinning',
        target_png_dir='/home/aljadaaa/Documents/skelnton/Road_Skeletonization/thinning_data/data/targets_png',
        geojson_dir = '/home/aljadaaa/Documents/skelnton/Road_Skeletonization/thinning_data/data/targets_geojson',
        transform=transform
    )

    total_size = len(dataset)
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Initialize model
    model = UNet(n_channels=1, n_classes=1).to(config.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Train
    #train_losses, val_losses = train_model(train_loader, val_loader, model, criterion, optimizer, config)
    """
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Progress')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    """
    
    # Evaluate
    model.load_state_dict(torch.load("unet_skeleton.pth", map_location=config.device))
    model.eval()
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
