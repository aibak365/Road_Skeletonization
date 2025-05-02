# Deep Learning-Based Learned Skeletonization of Road Networks

This project aims to reconstruct clean, one-pixel-wide skeleton representations from thick and noisy road images. It is part of our coursework for CSE627 - Spring 2024–25, focusing on semantic segmentation using deep learning.

## 📌 Objective
To develop a U-Net-based model that can extract accurate road skeletons from distorted road network images generated using OpenStreetMap (OSM) data.

## 🧱 Model Architecture
- Based on U-Net with 5 encoder and 4 decoder levels.
- Enhanced with double convolutional blocks and skip connections.
- Final output: 1x256x256 binary segmentation mask.

## 🧪 Dataset and Preprocessing
- **Input**: Thickened 256×256 grayscale road images from OSM.
- **Target**: One-pixel-wide ground truth skeleton masks.
- **Distortions Applied**: Gaussian noise, salt & pepper noise, motion blur, and defocus blur.
- Dataset prepared using a custom `RoadSkeletonDataset` class.

## ⚙️ Training Details
- **Loss Functions**: BCEWithLogitsLoss, Dice Loss, and a Combined Loss (weighted sum of both).
- **Optimizers**: Adam (lr = 1e-4 to 1e-5).
- **Batch Size**: 8
- **Epochs**: Up to 250
- **Validation Strategy**: 70/15/15 train-val-test split with early stopping and learning rate scheduling.

## 📈 Evaluation Metrics
- **Pixel-Level**: Test loss, MSE, IoU, Dice coefficient.
- **Node-Level**: Precision & Recall for valence-1 to valence-4 nodes using bipartite matching.

## 🔬 Ablation Study
We compared performance using different loss functions and training durations:
- Loss types: BCE, Dice, Combined (BCE + Dice)
- Epochs: 60, 80, 180, and 250
- Observed that Combined Loss at 180+ epochs provided best trade-off between detail preservation and generalization.

## 🔍 Results
- Best Dice Score: 0.8703
- Best IoU: 0.7894
- Strong node-level recall across valence types, especially valence-2 and valence-4 junctions.
- Visual inspections confirm the continuity and thinness of the predicted skeletons.

## 📁 Repository Structure
```
├── dataset_preperation.py        # Dataset loading and distortion augmentation
├── model_train_eval.py           # Training, evaluation, and loss comparison
├── model.py                      # U-Net architecture
├── README.md                     # This file
└── sample_prediction.png         # Example output image (input, pred, ground truth)
```

## 🤝 Team Members
- Md Aibak Aljadayah  
- Md Nahid Hasan  
- Md Nadim Mahmud  

## 🔗 GitHub Repo
https://github.com/aibak365/Road_Skeletonization
