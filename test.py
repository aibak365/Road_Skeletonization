import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from model_train_eval import UNet, evaluate_model
from dataset_preperation import RoadSkeletonDataset
from sklearn.metrics import jaccard_score
def load_and_test(model_path, test_data_dir, config):
    # 1. Initialize Model Architecture
    model = UNet(n_channels=1, n_classes=1).to(config.device)
    
    # 2. Load Saved Weights
    model.load_state_dict(torch.load(model_path, map_location=config.device, weights_only=True))
    model.eval()  # Set to evaluation mode
    
    # 3. Prepare Test Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    test_dataset = RoadSkeletonDataset(
        thinning_dir=test_data_dir['thinning'],
        target_png_dir=test_data_dir['targets'],
        geojson_dir=test_data_dir['geojson'],
        transform=transform,
        apply_distortions=False  # Disable distortions for testing
    )
    
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    # 4. Run Evaluation
    metrics = evaluate_model(test_loader, model, config)
    
    # 5. Visualize Predictions
    visualize_predictions(model, test_dataset, config, num_samples=3)
    
    return metrics

def visualize_predictions(model, dataset, config, num_samples=3):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    plt.figure(figsize=(15, 5*num_samples))
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        with torch.no_grad():
            input_tensor = sample['input'].unsqueeze(0).float().to(config.device)
            pred = torch.sigmoid(model(input_tensor)).cpu().numpy()
        
        plt.subplot(num_samples, 3, i*3+1)
        plt.imshow(sample['input'].squeeze(), cmap='gray')
        plt.title(f'Input {i+1}')
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3+2)
        plt.imshow(pred.squeeze() > config.threshold, cmap='gray')
        plt.title(f'Prediction {i+1}')
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3+3)
        plt.imshow(sample['target'].squeeze(), cmap='gray')
        plt.title(f'Ground Truth {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_predictions.png')
    plt.show()

# Configuration (use same as training)
class Config:
    def __init__(self):
        self.batch_size = 8
        self.threshold = 0.5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_path = "unet_skeleton.pth"

if __name__ == "__main__":
    config = Config()
    
    # Path configuration
    test_data_paths = {
        'thinning': '/home/aljadaaa/Documents/skelnton/Road_Skeletonization/thinning_data/data/thinning',
        'targets': '/home/aljadaaa/Documents/skelnton/Road_Skeletonization/thinning_data/data/targets_png',
        'geojson': '/home/aljadaaa/Documents/skelnton/Road_Skeletonization/thinning_data/data/targets_geojson'
    }
    
    # Load and test
    metrics = load_and_test(config.model_save_path, test_data_paths, config)
    
    print("\nFinal Test Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
