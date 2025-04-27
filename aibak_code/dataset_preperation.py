import os
from PIL import Image
import torch

class RoadSkeletonDataset(torch.utils.data.Dataset):
    def __init__(self, thinning_dir, target_png_dir, geojson_dir, transform=None, apply_distortions=True):
        self.thinning_dir = thinning_dir
        self.target_png_dir = target_png_dir
        self.geojson_dir = geojson_dir
        self.transform = transform
        self.apply_distortions = apply_distortions
        
        self.indices = [
            fname.split('_')[-1].split('.')[0]
            for fname in sorted(os.listdir(thinning_dir))
            if fname.endswith('.png')
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        thick_path = os.path.join(self.thinning_dir, f'image_{index}.png')  # Input thick road
        skeleton_path = os.path.join(self.target_png_dir, f'target_{index}.png')  # Ground truth skeleton
        geojson_path = os.path.join(self.geojson_dir, f'target_{index}.geojson')

        # Load images with corrected roles
        thick_img = np.array(Image.open(thick_path).convert('L'))  # Input thick road
        skeleton_img = np.array(Image.open(skeleton_path).convert('L'))  # Ground truth skeleton
        
        # Apply distortions to the INPUT image (thick road)
        if self.apply_distortions:
            thick_img = add_distortions(thick_img)
        
        # Apply transforms if any
        if self.transform:
            thick_img = self.transform(thick_img)
            skeleton_img = self.transform(skeleton_img)
        if len(skeleton_img.shape) == 2:
            skeleton_img = torch.from_numpy(skeleton_img).unsqueeze(0).cpu().detach().numpy()  # Add channel dimension
        if len(thick_img.shape) == 2:
            thick_img = torch.from_numpy(thick_img).unsqueeze(0).cpu().detach().numpy()
        return {
            'input': thick_img,  # Input to model (thick road)
            'target': skeleton_img,  # Ground truth (thin skeleton)
            'index': index,
            # 'geojson': geojson_data,  # Uncomment if needed
        }

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
from scipy import ndimage

def add_noise(image, noise_type='gaussian'):
    """Add various types of noise to the image"""
    # Convert to float for processing
    img = image.astype(np.float32) / 255.0
    
    if noise_type == 'gaussian':
        # Gaussian noise with random standard deviation
        sigma = random.uniform(0.01, 0.05)
        noise = np.random.normal(0, sigma, img.shape)
        noisy = img + noise
    
    elif noise_type == 'salt_pepper':
        # Salt and pepper noise
        s_vs_p = 0.5  # ratio of salt vs. pepper
        amount = random.uniform(0.01, 0.1)
        noisy = np.copy(img)
        
        # Salt (white) noise
        salt_mask = np.random.random(img.shape) < amount * s_vs_p
        noisy[salt_mask] = 1
        
        # Pepper (black) noise
        pepper_mask = np.random.random(img.shape) < amount * (1 - s_vs_p)
        noisy[pepper_mask] = 0
    
    elif noise_type == 'speckle':
        # Speckle noise (multiplicative noise)
        sigma = random.uniform(0.05, 0.15)
        noise = np.random.normal(0, sigma, img.shape)
        noisy = img + img * noise
        
    elif noise_type == 'poisson':
        # Poisson noise (simulates photon counting noise)
        noisy = np.random.poisson(img * 255) / 255
    
    # Clip values to valid range and convert back to original dtype
    noisy = np.clip(noisy, 0, 1) * 255
    return noisy.astype(np.uint8)

def add_blur(image, blur_type='gaussian'):
    """Add various types of blur to the image"""
    # Convert to float for processing
    img = image.astype(np.float32) / 255.0
    
    if blur_type == 'gaussian':
        # Gaussian blur with random sigma
        sigma = random.uniform(0.5, 2.0)
        blurred = ndimage.gaussian_filter(img, sigma)
    
    elif blur_type == 'motion':
        # Motion blur - simulates camera/object movement
        size = random.choice([3, 5, 7, 9])
        angle = random.uniform(0, 360)
        # Create motion blur kernel
        kernel = np.zeros((size, size))
        center = size // 2
        
        if 0 <= angle < 45 or 135 <= angle < 225 or 315 <= angle <= 360:
            # Horizontal-ish blur
            kernel[center, :] = 1
        else:
            # Vertical-ish blur
            kernel[:, center] = 1
            
        # Normalize kernel
        kernel = kernel / np.sum(kernel)
        blurred = ndimage.convolve(img, kernel)
    
    elif blur_type == 'defocus':
        # Defocus blur - simulates out-of-focus imagery
        radius = random.choice([2, 3, 4, 5])
        
        # Generate disk-shaped kernel
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x**2 + y**2 <= radius**2
        kernel = np.zeros((2*radius+1, 2*radius+1))
        kernel[mask] = 1
        kernel = kernel / np.sum(kernel)
        
        blurred = ndimage.convolve(img, kernel)
    
    # Convert back to original dtype
    blurred = np.clip(blurred, 0, 1) * 255
    return blurred.astype(np.uint8)

def add_distortions(image, noise_prob=0.8, blur_prob=0.6):
    """
    Add realistic distortions to the input image
    
    Args:
        image: Input image (numpy array)
        noise_prob: Probability of adding noise
        blur_prob: Probability of adding blur
    
    Returns:
        Distorted image
    """
    # Convert to float for processing
    img = image.astype(np.float32) / 255.0
    
    # Potentially add noise
    if random.random() < noise_prob:
        noise_type = random.choice(['gaussian', 'salt_pepper', 'poisson', 'speckle'])
        img = add_noise(img * 255, noise_type) / 255.0
    
    # Potentially add blur
    if random.random() < blur_prob:
        blur_type = random.choice(['gaussian', 'motion', 'defocus'])
        img = add_blur(img * 255, blur_type) / 255.0
    
    # Clip values to valid range and convert back to original dtype
    img = np.clip(img, 0, 1) * 255
    return img.astype(np.uint8)

def visualize_dataset_samples(thinning_dir, target_png_dir, num_samples=3):
    """
    Visualize samples from the dataset with original and distorted versions
    
    Args:
        thinning_dir: Directory containing thick road images (inputs)
        target_png_dir: Directory containing thin skeleton images (ground truth)
        num_samples: Number of samples to visualize
    """
    # Get sorted list of indices based on thinning folder
    indices = []
    for file in sorted(os.listdir(thinning_dir)):
        if file.endswith('.png'):
            # Extract the index from image_00000.png format
            idx = file.split('_')[-1].split('.')[0]
            indices.append(idx)
    
    # Create a figure with subplots
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axs = np.array([axs])  # Make it indexable for the single sample case
    
    for i in range(min(num_samples, len(indices))):
        idx = indices[i]
        
        # Construct file paths with the correct naming format
        thick_path = os.path.join(thinning_dir, f'image_{idx}.png')  # Input thick road
        skeleton_path = os.path.join(target_png_dir, f'target_{idx}.png')  # Ground truth skeleton
        
        # Check if files exist
        if not os.path.exists(thick_path) or not os.path.exists(skeleton_path):
            print(f"Warning: Missing files for index {idx}")
            continue
        
        # Load images with corrected roles
        thick_img = np.array(Image.open(thick_path).convert('L'))  # Input thick road
        skeleton_img = np.array(Image.open(skeleton_path).convert('L'))  # Ground truth skeleton
        
        # Apply distortions to the INPUT image (thick road)
        distorted_img = add_distortions(thick_img)
        
        # Plot images with correct labels
        if num_samples == 1:
            axs[0].imshow(thick_img, cmap='gray')
            axs[0].set_title('Original Input (Thick Road)')
            axs[0].axis('off')
            
            axs[1].imshow(distorted_img, cmap='gray')
            axs[1].set_title('Distorted Input (Thick Road)')
            axs[1].axis('off')
            
            axs[2].imshow(skeleton_img, cmap='gray')
            axs[2].set_title('Ground Truth (Skeleton)')
            axs[2].axis('off')
        else:
            axs[i, 0].imshow(thick_img, cmap='gray')
            axs[i, 0].set_title(f'Original Input #{i+1}')
            axs[i, 0].axis('off')
            
            axs[i, 1].imshow(distorted_img, cmap='gray')
            axs[i, 1].set_title(f'Distorted Input #{i+1}')
            axs[i, 1].axis('off')
            
            axs[i, 2].imshow(skeleton_img, cmap='gray')
            axs[i, 2].set_title(f'Ground Truth #{i+1}')
            axs[i, 2].axis('off')


def test_distortion_grid(image_path, num_rows=2, num_cols=3):
    """
    Create a grid showing different distortion types on a single image
    
    Args:
        image_path: Path to input image
        num_rows: Number of rows in the grid
        num_cols: Number of columns in the grid
    """
    # Load the image
    input_img = np.array(Image.open(image_path).convert('L'))
    
    # Create a figure with subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*4, num_rows*4))
    axs = axs.flatten()
    
    # Show original image
    axs[0].imshow(input_img, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    # Add different distortion types
    distortion_types = [
        ('Gaussian Noise', lambda img: add_noise(img, 'gaussian')),
        ('Salt & Pepper Noise', lambda img: add_noise(img, 'salt_pepper')),
        ('Gaussian Blur', lambda img: add_blur(img, 'gaussian')),
        ('Motion Blur', lambda img: add_blur(img, 'motion')),
        ('Combined (Noise+Blur)', lambda img: add_distortions(img))
    ]
    
    for i, (title, distort_func) in enumerate(distortion_types):
        if i+1 < len(axs):
            distorted = distort_func(input_img)
            axs[i+1].imshow(distorted, cmap='gray')
            axs[i+1].set_title(title)
            axs[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Directory paths - adjust to your folder locations
    thinning_dir = '/home/aljadaaa/Documents/skelnton/Road_Skeletonization/thinning_data/data/thinning'
    target_png_dir = '/home/aljadaaa/Documents/skelnton/Road_Skeletonization/thinning_data/data/targets_png'
    
    # Test 1: Visualize multiple dataset samples
    print("Visualizing dataset samples with distortions...")
    visualize_dataset_samples(thinning_dir, target_png_dir, num_samples=3)
    
    # Test 2: Show different distortion types on a single image
    indices = [
        file.split('_')[-1].split('.')[0] 
        for file in sorted(os.listdir(thinning_dir)) 
        if file.endswith('.png')
    ]
    
    if indices:
        print("Showing distortion grid for a single sample...")
        idx = indices[0]
        thick_path = os.path.join(thinning_dir, f'image_{idx}.png')  # Input thick road
        test_distortion_grid(thick_path)




