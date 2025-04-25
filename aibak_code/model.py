import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics import Accuracy, Precision, Recall, F1Score

import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
class RoadSkeletonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Map OSM highway types to numerical values
        self.highway_types = {
            'motorway': 0, 'trunk': 1, 'primary': 2,
            'secondary': 3, 'tertiary': 4, 'residential': 5,
            'service': 6, 'footway': 7, 'cycleway': 8
        }
        
        # Get paired samples
        self.samples = []
        thinning_dir = self.root_dir / 'thinning'
        for img_path in thinning_dir.glob('*.png'):
            stem = img_path.stem
            target_path = self.root_dir / 'targets_png' / f'{stem}.png'
            geojson_path = self.root_dir / 'targets_geojson' / f'{stem}.geojson'
            
            if target_path.exists() and geojson_path.exists():
                self.samples.append((img_path, target_path, geojson_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target_path, geojson_path = self.samples[idx]
        
        # Load images
        image = Image.open(img_path).convert('L')
        target = Image.open(target_path).convert('L')
        
        # Threshold target to strict binary values (0 or 255)
        target = target.point(lambda x: 0 if x < 128 else 255)
        
        # Process GeoJSON features
        with open(geojson_path) as f:
            geojson = json.load(f)
        properties = geojson['features'][0]['properties']
        lanes = int(properties.get('lanes', 1))
        highway_type = self.highway_types.get(
            properties.get('highway', 'residential'), 5
        )
        
        # Apply transforms to image only
        if self.transform:
            image = self.transform(image)  # Includes normalization
            
        # Convert target to tensor (automatically scales 255->1.0)
        target = transforms.ToTensor()(target)
        
        # Verify binary values
        assert torch.all(torch.logical_or(target == 0.0, target == 1.0)), \
            "Target contains invalid values"
            
        return image, target, {
            'lanes': torch.tensor(lanes),
            'highway_type': torch.tensor(highway_type)
        }

    
train_transform = transforms.Compose([
    transforms.RandomRotation([90, 90]),  # From paper §3.3
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

#root_path = '/home/aljadaaa/Documents/skelnton/Road_Skeletonization/thinning_data/data'
#train_dataset = RoadSkeletonDataset(root_dir=root_path, transform=train_transform)




def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    annotations = [item[2] for item in batch]
    return images, targets, annotations


root_path = '/home/aljadaaa/Documents/skelnton/Road_Skeletonization/thinning_data/data'
train_dataset = RoadSkeletonDataset(
    root_dir=root_path,
    transform=train_transform
)


import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=5):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResidualBlock(in_channels) for _ in range(num_blocks)]
        )
        self.pool = nn.MaxPool2d(2)
        self.proj = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.blocks(x)
        skip = self.proj(x)  # Project for skip connection
        x = self.pool(skip)  # Pool the projected output, not the original
        return x, skip

    """def forward(self, x):
        x = self.blocks(x)
        skip = self.proj(x)
        x = self.pool(x)
        return x, skip"""

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.skip_conv = nn.Conv2d(skip_channels, out_channels, kernel_size=1)  # New projection layer
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x, skip):
        x = self.up(x)
        skip = self.skip_conv(skip)  # Project skip to match channels
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SkeletonUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Downsampling path
        self.down1 = DownBlock(in_channels, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.down5 = DownBlock(512, 1024)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU()
        )
        
        # Upsampling path
        self.up1 = UpBlock(1024, 512, skip_channels=1024)
        self.up2 = UpBlock(512, 256, skip_channels=512)
        self.up3 = UpBlock(256, 128, skip_channels=256)
        self.up4 = UpBlock(128, 64, skip_channels=128)
        self.up5 = UpBlock(64, 32, skip_channels=64)
        
        # Final output
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        x, skip5 = self.down5(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.up1(x, skip5)
        x = self.up2(x, skip4)
        x = self.up3(x, skip3)
        x = self.up4(x, skip2)
        x = self.up5(x, skip1)
        
        # Final output
        x = self.final(x)
        return self.sigmoid(x)



# Weighted Focal Loss implementation
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        """
        Proper Focal Loss implementation for binary classification
        Args:
            alpha (float): Weighting factor for positive class (range 0-1)
            gamma (float): Focusing parameter (>=0)
            reduction (str): 'mean', 'sum' or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure inputs are probabilities (after sigmoid)
        inputs = inputs.float()
        targets = targets.float()
        
        # Calculate binary cross entropy
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Calculate p_t
        pt = torch.exp(-BCE_loss)  # p if target=1, 1-p if target=0
        
        # Create alpha_t tensor
        alpha_t = torch.ones_like(targets) * (1 - self.alpha)
        alpha_t[targets == 1] = self.alpha  # Apply alpha to positive class
        
        # Calculate focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

        # Handle reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# Initialize model and optimizer
model = SkeletonUNet()
criterion = WeightedFocalLoss(alpha=50, gamma=2)  # From paper [1]
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # From [3]

# Data Augmentation (from paper [1])
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=(90, 270)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, threshold=0.5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.threshold = threshold
        
        # Initialize metrics storage
        self.metrics = {
            'train_metrics': {
                'loss': [], 'acc': [], 'prec': [], 'rec': [], 'f1': []
            },
            'val_metrics': {
                'loss': [], 'acc': [], 'prec': [], 'rec': [], 'f1': []
            }
        }
        
        # Initialize metric modules
        self.train_metrics = nn.ModuleDict({
            'acc': Accuracy(task='binary').to(device),
            'prec': Precision(task='binary').to(device),
            'rec': Recall(task='binary').to(device),
            'f1': F1Score(task='binary').to(device)
        })
        
        self.val_metrics = nn.ModuleDict({
            'acc': Accuracy(task='binary').to(device),
            'prec': Precision(task='binary').to(device),
            'rec': Recall(task='binary').to(device),
            'f1': F1Score(task='binary').to(device)
        })
        
        # Loss and optimizer
        self.criterion = WeightedFocalLoss(alpha=50, gamma=2)
        self.optimizer = optim.Adam(model.parameters(), lr=3e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', patience=3, factor=0.5)

    def _compute_metrics(self, preds, targets, mode='train'):
        """Calculate batch metrics"""
        with torch.no_grad():
            bin_preds = (preds > self.threshold).float()
            metrics = self.train_metrics if mode == 'train' else self.val_metrics
            
            acc = metrics['acc'](bin_preds, targets)
            prec = metrics['prec'](bin_preds, targets)
            rec = metrics['rec'](bin_preds, targets)
            f1 = metrics['f1'](bin_preds, targets)
            
        return acc, prec, rec, f1

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        for metric in self.train_metrics.values():
            metric.reset()
            
        for inputs, targets, _ in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()

        # Compute epoch metrics
        with torch.no_grad():
            self.model.eval()
            acc, prec, rec, f1 = self._compute_metrics(outputs, targets, 'train')
            self.model.train()
            
        # Store metrics
        n = len(self.train_loader)
        self.metrics['train_metrics']['loss'].append(epoch_loss/n)
        self.metrics['train_metrics']['acc'].append(acc.item())
        self.metrics['train_metrics']['prec'].append(prec.item())
        self.metrics['train_metrics']['rec'].append(rec.item())
        self.metrics['train_metrics']['f1'].append(f1.item())

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        epoch_loss = 0
        for metric in self.val_metrics.values():
            metric.reset()
            
        for inputs, targets, _ in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            epoch_loss += loss.item()
            
            # Update metrics
            self.val_metrics['acc'].update(outputs, targets)
            self.val_metrics['prec'].update(outputs, targets)
            self.val_metrics['rec'].update(outputs, targets)
            self.val_metrics['f1'].update(outputs, targets)

        # Compute final metrics
        final_acc = self.val_metrics['acc'].compute().item()
        final_prec = self.val_metrics['prec'].compute().item()
        final_rec = self.val_metrics['rec'].compute().item()
        final_f1 = self.val_metrics['f1'].compute().item()

        # Store metrics
        n = len(self.val_loader)
        self.metrics['val_metrics']['loss'].append(epoch_loss/n)
        self.metrics['val_metrics']['acc'].append(final_acc)
        self.metrics['val_metrics']['prec'].append(final_prec)
        self.metrics['val_metrics']['rec'].append(final_rec)
        self.metrics['val_metrics']['f1'].append(final_f1)

        return final_f1

    def fit(self, epochs, early_stop=5):
        best_f1 = 0
        no_improve = 0
        
        for epoch in range(epochs):
            self.train_epoch()
            val_f1 = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_f1)
            
            # Early stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                no_improve = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                no_improve += 1
                
            if no_improve >= early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
            self._print_metrics(epoch+1)

    def _print_metrics(self, epoch):
    #"""Print metrics for current epoch"""
        train = self.metrics['train_metrics']  # Changed from 'train'
        val = self.metrics['val_metrics']      # Changed from 'val'

        print(f"Epoch {epoch}:")
        print(f"Train Loss: {train['loss'][-1]:.4f} | Acc: {train['acc'][-1]:.4f} | F1: {train['f1'][-1]:.4f}")
        print(f"Val   Loss: {val['loss'][-1]:.4f} | Acc: {val['acc'][-1]:.4f} | F1: {val['f1'][-1]:.4f}\n")



        
    def _compute_metrics(self, preds, targets, mode='train'):
        #"""Calculate batch metrics"""
        with torch.no_grad():
            bin_preds = (preds > self.threshold).float()
            
            if mode == 'train':
                metrics = self.train_metrics
            else:
                metrics = self.val_metrics
                
            acc = metrics['acc'](bin_preds, targets)
            prec = metrics['prec'](bin_preds, targets)
            rec = metrics['rec'](bin_preds, targets)
            f1 = metrics['f1'](bin_preds, targets)
            
        return acc, prec, rec, f1



    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        # Reset metrics at start of epoch
        for metric in self.train_metrics.values():
            metric.reset()
            
        for inputs, targets, _ in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            
            # Update metrics
            self.train_metrics['acc'].update(outputs, targets)
            self.train_metrics['prec'].update(outputs, targets)
            self.train_metrics['rec'].update(outputs, targets)
            self.train_metrics['f1'].update(outputs, targets)

        # Compute final metrics for the epoch
        final_acc = self.train_metrics['acc'].compute().item()
        final_prec = self.train_metrics['prec'].compute().item()
        final_rec = self.train_metrics['rec'].compute().item()
        final_f1 = self.train_metrics['f1'].compute().item()

        # Store averaged metrics
        n = len(self.train_loader)
        self.metrics['train_metrics']['loss'].append(epoch_loss/n)
        self.metrics['train_metrics']['acc'].append(final_acc)
        self.metrics['train_metrics']['prec'].append(final_prec)
        self.metrics['train_metrics']['rec'].append(final_rec)
        self.metrics['train_metrics']['f1'].append(final_f1)



    @torch.no_grad()
    def validate(self):
        self.model.eval()
        epoch_loss = 0
        
        # Reset metrics at start of validation
        for metric in self.val_metrics.values():
            metric.reset()
            
        for inputs, targets, _ in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            epoch_loss += loss.item()
            
            # Update metrics using ModuleDict
            bin_preds = (outputs > self.threshold).float()
            self.val_metrics['acc'].update(bin_preds, targets)
            self.val_metrics['prec'].update(bin_preds, targets)
            self.val_metrics['rec'].update(bin_preds, targets)
            self.val_metrics['f1'].update(bin_preds, targets)

        # Compute final metrics
        final_acc = self.val_metrics['acc'].compute().item()
        final_prec = self.val_metrics['prec'].compute().item()
        final_rec = self.val_metrics['rec'].compute().item()
        final_f1 = self.val_metrics['f1'].compute().item()

        # Store metrics with proper averaging
        n = len(self.val_loader)
        self.metrics['val_metrics']['loss'].append(epoch_loss/n)
        self.metrics['val_metrics']['acc'].append(final_acc)
        self.metrics['val_metrics']['prec'].append(final_prec)
        self.metrics['val_metrics']['rec'].append(final_rec)
        self.metrics['val_metrics']['f1'].append(final_f1)

        # Reset metrics for next validation
        for metric in self.val_metrics.values():
            metric.reset()

        return final_f1

    

    def fit(self, epochs, early_stop=5):
        best_f1 = 0
        no_improve = 0
        
        for epoch in range(epochs):
            self.train_epoch()
            val_f1 = self.validate()
            
            # Update scheduler (paper §4)
            self.scheduler.step(val_f1)
            
            # Early stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                no_improve = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                no_improve += 1
                
            if no_improve >= early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
            self._print_metrics(epoch+1)


class Tester:
    def __init__(self, model, test_loader, device, threshold=0.5):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.threshold = threshold
        
        # Metrics
        self.metrics = {
            'acc': 0, 'prec': 0, 'rec': 0, 'f1': 0
        }
        
    def test(self):
        self.model.eval()
        acc_meter, prec_meter, rec_meter, f1_meter = 0, 0, 0, 0
        
        with torch.no_grad():
            for inputs, targets, _ in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                bin_preds = (outputs > self.threshold).float()
                
                # Update metrics
                acc_meter += Accuracy(task='binary')(bin_preds, targets).item()
                prec_meter += Precision(task='binary')(bin_preds, targets).item()
                rec_meter += Recall(task='binary')(bin_preds, targets).item()
                f1_meter += F1Score(task='binary')(bin_preds, targets).item()
                
        n = len(self.test_loader)
        self.metrics['acc'] = acc_meter/n
        self.metrics['prec'] = prec_meter/n
        self.metrics['rec'] = rec_meter/n
        self.metrics['f1'] = f1_meter/n
        
        return self.metrics
    
    def visualize_samples(self, num_samples=3):
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        self.model.eval()
        
        with torch.no_grad():
            for i, (inputs, targets, _) in enumerate(self.test_loader):
                if i >= num_samples:
                    break
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds = (outputs > self.threshold).float().cpu()
                
                # Denormalize
                img = inputs[0].cpu().numpy().squeeze()
                img = (img * 0.5 + 0.5) * 255
                
                axes[i,0].imshow(img, cmap='gray')
                axes[i,0].set_title('Input')
                axes[i,1].imshow(targets[0].squeeze(), cmap='gray')
                axes[i,1].set_title('Ground Truth')
                axes[i,2].imshow(preds[0].squeeze(), cmap='gray')
                axes[i,2].set_title('Prediction')
                
        plt.tight_layout()
        plt.savefig('test_samples.png')
        plt.close()

# Usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SkeletonUNet()

# Add this after your dataset class definition but before model initialization

# Split dataset into train/val/test (paper used 80:20 split)
total_samples = len(train_dataset)
train_size = int(0.8 * total_samples)
val_size = int(0.1 * total_samples)
test_size = total_samples - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)

# Create dataloaders
batch_size = 16  # From paper's implementation
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

# Verify the splits
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Total samples: {len(train_dataset)+len(val_dataset)+len(test_dataset)}")

train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=(90, 90)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # For inputs only
])
from torch.utils.data import DataLoader

batch_size = 16  # As used in the paper or adjust for your GPU

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SkeletonUNet().to(device)
criterion = WeightedFocalLoss(alpha=50, gamma=2)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    threshold=0.5  # Use 0.5 for training/validation, 0.81 for final test as in the paper
)

trainer.fit(epochs=100, early_stop=5)
tester = Tester(model, test_loader, device, threshold=0.81)  # Paper's threshold for binarization
metrics = tester.test()
print(f"Test Metrics: {metrics}")
tester.visualize_samples()