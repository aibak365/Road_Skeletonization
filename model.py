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

class ThinningDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.startswith("image_") and f.endswith(".png")])
        self.target_files = sorted([f for f in os.listdir(root_dir) if f.startswith("target_") and f.endswith(".png") and f.replace("target_", "image_") in self.image_files])
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        tgt_path = os.path.join(self.root_dir, self.target_files[idx])
        image = Image.open(img_path).convert("L")
        target = Image.open(tgt_path).convert("L")
        return self.transform(image), self.transform(target), img_path, tgt_path

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

def train_model(model, train_loader, device, epochs=20, lr=1e-4):
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

def visualize_prediction(image, pred, target, idx):
    image = image.squeeze().cpu().numpy()
    pred = pred.squeeze().detach().cpu().numpy()
    target = target.squeeze().cpu().numpy()
    plt.subplot(5, 3, 3*idx + 1)
    plt.imshow(image, cmap='gray'); plt.title("Input"); plt.axis('off')
    plt.subplot(5, 3, 3*idx + 2)
    plt.imshow(pred, cmap='gray'); plt.title("Prediction"); plt.axis('off')
    plt.subplot(5, 3, 3*idx + 3)
    plt.imshow(target, cmap='gray'); plt.title("Ground Truth"); plt.axis('off')

def run_pipeline(data_dir="thinning_data/data/thinning"):
    dataset = ThinningDataset(data_dir)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = UNet()
    
    train_model(model, train_loader, device)

    # Visualization
    model.load_state_dict(torch.load("unet_skeleton.pth", map_location=device))
    model.eval()
    plt.figure(figsize=(12, 20))
    with torch.no_grad():
        for idx, (img, tgt, _, _) in enumerate(val_loader):
            if idx >= 5: break
            img, tgt = img.to(device), tgt.to(device)
            pred = model(img)
            visualize_prediction(img, pred, tgt, idx)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_pipeline()