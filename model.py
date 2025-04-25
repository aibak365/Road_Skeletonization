import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from collections import defaultdict
from scipy.ndimage import distance_transform_edt, convolve
from scipy.spatial import cKDTree
import random
import scipy

class ThinningDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.startswith("image_") and f.endswith(".png")])
        self.target_files = sorted([f for f in os.listdir(root_dir) if f.startswith("target_") and f.endswith(".png") and f.replace("target_", "image_") in self.image_files])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        tgt_path = os.path.join(self.root_dir, self.target_files[idx])
        image = Image.open(img_path).convert("L")
        target = Image.open(tgt_path).convert("L")
        
        # added transform
        if self.transform:
            image = self.transform(image)
            target = self.transform(target)
        else:
            image = transforms.ToTensor()(image)
            target = transforms.ToTensor()(target)

        return image, target, img_path, tgt_path

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        
        self.enc1 = nn.Sequential(CBR(1, 64), CBR(64, 64))
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.enc4 = nn.Sequential(CBR(256, 512), CBR(512, 512))

        self.pool = nn.MaxPool2d(2)
        
        self.middle = nn.Sequential(CBR(512, 1024), CBR(1024, 512))
        self.upconv4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec4 = nn.Sequential(CBR(1024, 512), CBR(512, 256))
        self.upconv3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec3 = nn.Sequential(CBR(512, 256), CBR(256, 128))
        self.upconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 64))
        self.upconv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        mid = self.middle(self.pool(e4))

        d4 = self.upconv4(mid)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.upconv3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.upconv2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return torch.sigmoid(self.final(d1))
    
class MinMaxNormalize:
    def __call__(self, tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    
def load_datsets(data_dir, test_split=0.2):

    transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        MinMaxNormalize(),
    ])

    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        MinMaxNormalize(),
    ])

    full_dataset = ThinningDataset(data_dir, transform=None)

    val_size = int(test_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    indices = torch.randperm(len(full_dataset)).tolist()
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_dataset = torch.utils.data.Subset(ThinningDataset(data_dir, transform=transform_train), train_indices)
    val_dataset = torch.utils.data.Subset(ThinningDataset(data_dir, transform=transforms_test), val_indices)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader


def train_model(model, train_loader, device, epochs=1, lr=1e-4):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, targets, _, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    
    torch.save(model.state_dict(), "unet_skeleton.pth")
    print("Model saved as 'unet_skeleton.pth'")

def predict_on_loader(model, loader, device, num_samples=5):
    model.eval()
    predictions = []
    with torch.no_grad():
        for idx, (img, tgt, _, _) in enumerate(loader):
            if idx >= num_samples: break
            img, tgt = img.to(device), tgt.to(device)
            pred = model(img)
            predictions.append((img, pred, tgt))
    return predictions

def visualize_predictions(predictions):
    plt.figure(figsize=(12, len(predictions) * 4))
    for idx, (img, pred, tgt) in enumerate(predictions):
        image = img.squeeze().cpu().numpy()
        pred = pred.squeeze().cpu().numpy()
        target = tgt.squeeze().cpu().numpy()

        plt.subplot(len(predictions), 3, idx*3 + 1)
        plt.imshow(image, cmap='gray')
        plt.title("Input")
        plt.axis('off')

        plt.subplot(len(predictions), 3, idx*3 + 2)
        plt.imshow(pred, cmap='gray')
        plt.title("Prediction")
        plt.axis('off')

        plt.subplot(len(predictions), 3, idx*3 + 3)
        plt.imshow(target, cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def qualitative_and_quantitative_evaluation(model, val_loader, device, num_visuals=5):
    model.eval()
    loss_fn = torch.nn.BCELoss()
    total_loss = 0
    dice_scores = []
    mse_scores = []
    precision_scores = defaultdict(list)
    recall_scores = defaultdict(list)
    visuals = []

    def dice_coefficient(pred, target, threshold=0.5):
        pred_bin = (pred > threshold).float()
        intersection = (pred_bin * target).sum()
        return (2. * intersection) / (pred_bin.sum() + target.sum() + 1e-8)

    def mean_squared_error_from_distance(pred, target):
        target_bin = target.squeeze().cpu().numpy() > 0.5
        pred_bin = pred.squeeze().cpu().numpy() > 0.5
        dt = distance_transform_edt(~target_bin)
        mse = np.mean(dt[pred_bin])
        return mse

    def node_valences(skeleton):
        from scipy.ndimage import convolve
        kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
        filtered = convolve(skeleton.astype(np.uint8), kernel, mode='constant')
        node_map = defaultdict(list)
        for y in range(1, skeleton.shape[0] - 1):
            for x in range(1, skeleton.shape[1] - 1):
                if skeleton[y, x]:
                    valence = (filtered[y, x] - 10)
                    if 1 <= valence <= 4:
                        node_map[valence].append((y, x))
        return node_map

    def bipartite_precision_recall(pred, target, valence, radius=3):
        pred_nodes = node_valences(pred).get(valence, [])
        gt_nodes = node_valences(target).get(valence, [])
        if not gt_nodes:
            return 0.0, 0.0
        tree = cKDTree(gt_nodes)
        matched = tree.query_ball_point(pred_nodes, r=radius)
        tp = sum(1 for m in matched if m)
        fp = len(pred_nodes) - tp
        fn = len(gt_nodes) - tp
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        return precision, recall

    with torch.no_grad():
        for idx, (img, tgt, _, _) in enumerate(tqdm(val_loader, desc="Evaluating")):
            img, tgt = img.to(device), tgt.to(device)
            pred = model(img)
            loss = loss_fn(pred, tgt)
            total_loss += loss.item()

            dice_scores.append(dice_coefficient(pred, tgt).item())
            mse_scores.append(mean_squared_error_from_distance(pred, tgt))

            pred_bin = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            tgt_bin = (tgt.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

            for v in [1, 2, 3, 4]:
                p, r = bipartite_precision_recall(pred_bin, tgt_bin, valence=v)
                precision_scores[v].append(p)
                recall_scores[v].append(r)

            if idx < num_visuals:
                visuals.append((img.squeeze().cpu().numpy(), pred.squeeze().cpu().numpy(), tgt.squeeze().cpu().numpy()))

    # Display qualitative results
    plt.figure(figsize=(12, 4 * num_visuals))
    for i, (img_np, pred_np, tgt_np) in enumerate(visuals):
        plt.subplot(num_visuals, 3, i * 3 + 1)
        plt.imshow(img_np, cmap='gray')
        plt.title("Input")
        plt.axis('off')

        plt.subplot(num_visuals, 3, i * 3 + 2)
        plt.imshow(pred_np, cmap='gray')
        plt.title("Prediction")
        plt.axis('off')

        plt.subplot(num_visuals, 3, i * 3 + 3)
        plt.imshow(tgt_np, cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Print quantitative metrics
    print("\n Quantitative Evaluation:")
    print(f"Test Loss (BCE): {total_loss / len(val_loader):.4f}")
    print(f"Dice Coefficient: {np.mean(dice_scores):.4f}")
    print(f"MSE (from distance transform): {np.mean(mse_scores):.4f}")
    for v in [1, 2, 3, 4]:
        print(f"Valence {v} - Precision: {np.mean(precision_scores[v]):.4f}, Recall: {np.mean(recall_scores[v]):.4f}")

    return {
        "loss": total_loss / len(val_loader),
        "dice": np.mean(dice_scores),
        "mse": np.mean(mse_scores),
        "valence_precision": {v: np.mean(precision_scores[v]) for v in [1, 2, 3, 4]},
        "valence_recall": {v: np.mean(recall_scores[v]) for v in [1, 2, 3, 4]},
    }

def run_pipeline(data_dir="thinning_data/data/thinning"):

    # Load datasets
    train_loader, val_loader = load_datsets(data_dir)


    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = UNet()

    # Train model
    train_model(model, train_loader, device, epochs=100, lr=1e-4)

    # Load trained model and evaluate
    model.load_state_dict(torch.load("unet_skeleton.pth", map_location=device))

    # Predict and visualize
    predictions = predict_on_loader(model, val_loader, device, num_samples=5)
    visualize_predictions(predictions)

#     results = qualitative_and_quantitative_evaluation(model, val_loader, device, num_visuals=5)

if __name__ == "__main__":
    run_pipeline()