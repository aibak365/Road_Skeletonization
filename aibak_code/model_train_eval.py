import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import os
from datetime import datetime
from tqdm import tqdm
from dataset_preperation import RoadSkeletonDataset
# Define the U-Net architecture
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet (encoder)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNet (decoder)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # Bottleneck layer
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse for the decoder path

        # Decoder path with skip connections
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            # Handle cases where dimensions don't match
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
                
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        # Final layer with sigmoid activation
        return torch.sigmoid(self.final_conv(x))

# Loss function: Dice Loss for segmentation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Flatten predictions and targets
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss
        return 1 - dice

# Combined loss: BCE + Dice
class BCEDiceLoss(nn.Module):
    def __init__(self, weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.weight = weight
        
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return bce_loss * self.weight + dice_loss * (1 - self.weight)

# Evaluation metrics
def calculate_mse(predicted, ground_truth):
    """Calculate MSE using distance transform from ground truth"""
    # Compute distance transform of ground truth
    dist_transform = distance_transform_edt(1 - ground_truth)
    
    # Calculate MSE
    mse = np.mean(np.square(predicted * dist_transform))
    return mse

def extract_nodes_by_valence(skeleton):
    """Extract nodes of different valences from a skeleton image"""
    # Pad the skeleton to handle edge cases
    padded = np.pad(skeleton, pad_width=1, mode='constant', constant_values=0)
    
    nodes = {v: [] for v in [1, 2, 3, 4]}
    
    # Find all non-zero points in the skeleton
    points = np.where(padded > 0)
    for y, x in zip(points[0], points[1]):
        # Skip edge pixels (from padding)
        if y == 0 or y == padded.shape[0]-1 or x == 0 or x == padded.shape[1]-1:
            continue
        
        # Count neighbors (3x3 neighborhood)
        neighbors = padded[y-1:y+2, x-1:x+2].copy()
        neighbors[1, 1] = 0  # Remove center pixel
        neighbor_count = np.sum(neighbors > 0)
        
        # Classify by valence
        if 0 < neighbor_count <= 4:
            nodes[neighbor_count].append((y-1, x-1))  # Adjust back for padding
    
    return nodes

def bipartite_match(pred_points, gt_points, max_distance=3):
    """Find matching points between prediction and ground truth"""
    if not pred_points or not gt_points:
        return []
        
    # Convert to numpy arrays
    pred_arr = np.array(pred_points)
    gt_arr = np.array(gt_points)
    
    # Calculate distance matrix
    dist_matrix = cdist(pred_arr, gt_arr)
    
    # Find matches
    matches = []
    matched_pred = set()
    matched_gt = set()
    
    while len(matched_pred) < len(pred_points) and len(matched_gt) < len(gt_points):
        # Find minimum distance
        min_dist = np.min(dist_matrix)
        if min_dist > max_distance:
            break
            
        # Find indices of minimum distance
        i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        
        if i not in matched_pred and j not in matched_gt:
            matches.append((i, j))
            matched_pred.add(i)
            matched_gt.add(j)
            
        # Set this distance to infinity to avoid reusing
        dist_matrix[i, j] = np.inf
        
    return matches

def calculate_node_metrics(predicted, ground_truth, max_distance=3):
    """Calculate precision and recall for nodes of different valences"""
    # Extract nodes of different valences
    pred_nodes = extract_nodes_by_valence(predicted)
    gt_nodes = extract_nodes_by_valence(ground_truth)
    
    results = {}
    
    # Calculate precision and recall for each valence
    for valence in [1, 2, 3, 4]:
        pred_points = pred_nodes[valence]
        gt_points = gt_nodes[valence]
        
        matches = bipartite_match(pred_points, gt_points, max_distance)
        
        precision = len(matches) / len(pred_points) if pred_points else 1.0
        recall = len(matches) / len(gt_points) if gt_points else 1.0
        
        results[f'valence_{valence}_precision'] = precision
        results[f'valence_{valence}_recall'] = recall
        
    return results

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25, patience=5):
    """Train the U-Net model"""
    model.to(device)
    best_loss = float('inf')
    best_model_weights = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    return model, history

# Evaluation function
def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data"""
    model.to(device)
    model.eval()
    
    # Metrics
    dice_loss = DiceLoss()
    test_loss = 0.0
    mse_values = []
    node_metrics_results = {val: {'precision': [], 'recall': []} for val in [1, 2, 3, 4]}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(inputs)
            loss = dice_loss(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            
            # Convert to numpy for metric calculation
            outputs_np = (outputs.cpu().numpy() > 0.5).astype(np.uint8)
            targets_np = (targets.cpu().numpy() > 0.5).astype(np.uint8)
            
            # Calculate metrics for each sample
            for i in range(outputs_np.shape[0]):
                # MSE using distance transform
                mse = calculate_mse(outputs_np[i, 0], targets_np[i, 0])
                mse_values.append(mse)
                
                # Node precision and recall
                metrics = calculate_node_metrics(outputs_np[i, 0], targets_np[i, 0])
                for valence in [1, 2, 3, 4]:
                    node_metrics_results[valence]['precision'].append(metrics[f'valence_{valence}_precision'])
                    node_metrics_results[valence]['recall'].append(metrics[f'valence_{valence}_recall'])
    
    # Calculate average metrics
    test_loss /= len(test_loader.dataset)
    avg_mse = np.mean(mse_values)
    
    results = {
        'test_loss': test_loss,
        'mse': avg_mse
    }
    
    # Average node metrics
    for valence in [1, 2, 3, 4]:
        results[f'valence_{valence}_precision'] = np.mean(node_metrics_results[valence]['precision'])
        results[f'valence_{valence}_recall'] = np.mean(node_metrics_results[valence]['recall'])
    
    return results

# Visualization function
def visualize_results(model, test_loader, device, num_samples=5, save_dir='results'):
    """Visualize model predictions"""
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        batch_count = 0
        for batch in test_loader:
            if batch_count >= num_samples:
                break
                
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(inputs)
            pred_thresh = (outputs > 0.5).float()
            
            # Convert to numpy
            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            outputs_np = pred_thresh.cpu().numpy()
            
            # Plot results
            for j in range(min(3, inputs.size(0))):  # Show at most 3 samples per batch
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(inputs_np[j, 0], cmap='gray')
                plt.title('Input (Thick Road)')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(outputs_np[j, 0], cmap='gray')
                plt.title('Prediction (Skeleton)')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(targets_np[j, 0], cmap='gray')
                plt.title('Ground Truth (Skeleton)')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'sample_{batch_count}_{j}.png'))
                plt.close()
            
            batch_count += 1

# Main function to run training and evaluation
def main(dataset, batch_size=16, val_split=0.2, test_split=0.1, 
         num_epochs=50, learning_rate=1e-4, output_dir='results'):
    """Main function to train and evaluate model"""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Split dataset into train, val, test
    total_size = len(dataset)
    test_size = int(test_split * total_size)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = UNet(in_channels=1, out_channels=1)
    
    # Define loss function and optimizer
    criterion = BCEDiceLoss(weight=0.3)  # Combine BCE and Dice losses
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train model
    print("Starting training...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs
    )
    
    # Save model
    model_path = os.path.join(output_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    # Visualize results
    print("Generating visualizations...")
    visualize_results(model, test_loader, device, save_dir=os.path.join(output_dir, "visualizations"))
    
    print(f"All results saved to {output_dir}")
    
    return model, metrics, history

# Example usage (to be called from another script)
if __name__ == "__main__":
    # This allows the file to be imported without running the main function
    thinning_dir = '/home/aljadaaa/Documents/skelnton/Road_Skeletonization/thinning_data/data/thinning'
    target_png_dir = '/home/aljadaaa/Documents/skelnton/Road_Skeletonization/thinning_data/data/targets_png'
    geo_dir = '/home/aljadaaa/Documents/skelnton/Road_Skeletonization/thinning_data/data/targets_geojson'
    main(RoadSkeletonDataset(thinning_dir, target_png_dir, geo_dir))
