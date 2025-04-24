import cv2

class RoadDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, target_paths, transform=None):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transform = transform
        
    def __getitem__(self, idx):
        # Load image and target
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(self.target_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Convert to tensors
        sample = {'image': image, 'target': target}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
