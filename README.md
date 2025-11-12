# EfficientNet-based Fault Detection

A PyTorch pipeline for **image-based fault detection** using a pre-trained **EfficientNet-B0**. This project handles **class imbalance**, applies **data augmentation**, and includes **training, validation, early stopping, and evaluation** with precision, recall, and F1-score metrics.  

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model Architecture](#model-architecture)  
- [Training](#training)  
- [Evaluation](#evaluation)  
- [Technologies Used](#technologies-used)  
- [License](#license)  

---

## Project Overview

This project implements a **deep learning pipeline** for detecting faults from images using **EfficientNet-B0**. It is designed to handle **imbalanced datasets**, ensure robust model performance through **data augmentation**, and evaluate predictions using **standard classification metrics**.  

---

## Features

- **Data Preprocessing & Augmentation:** Random crops, flips, rotation, color jitter, and normalization.  
- **Class Imbalance Handling:** Weighted random sampling and class-weighted loss function.  
- **Model Architecture:** Fine-tuning a pre-trained **EfficientNet-B0** for binary classification.  
- **Training Pipeline:**  
  - Adam optimizer with **ReduceLROnPlateau** scheduler  
  - **Early stopping** based on validation loss  
  - GPU acceleration if available  
- **Evaluation:** Generates **classification report** including precision, recall, and F1-score.  
- **Model Saving:** Saves the best-performing model as `best_model.pth`.  

---

## Dataset

- The dataset should be organized in the following structure:  

dataset/
├── train/
│ ├── faulty/
│ └── normal/
└── val/
├── faulty/
└── normal/


- Update the `train_path` and `val_path` variables in the script with your local dataset paths.  

---

## Installation

1. Clone the repository:  
```bash
git clone https://github.com/yourusername/fault-detection.git
cd fault-detection

 2. Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


3. Install dependencies:

pip install -r requirements.txt


4. Requirements:

Python >= 3.8

PyTorch

Torchvision

NumPy

scikit-learn

Usage

5. Set the paths for your training and validation datasets in the script:

train_path = r'path_to_train_folder'
val_path = r'path_to_val_folder'


6.Run the training script:

python main.py


The best model will be saved as best_model.pth.

7. Evaluate the model using the included classification report printed at the end of training.

8. Model Architecture

EfficientNet-B0 pre-trained on ImageNet

Fully connected layer replaced to output 2 classes

Trained with CrossEntropyLoss using class weights for imbalance

Training

Batch size: 32

Optimizer: Adam with learning rate 0.0003

Learning rate scheduler: ReduceLROnPlateau

Early stopping patience: 10 epochs

Maximum epochs: 40

Evaluation

Metrics: Validation Loss, Precision, Recall, F1-score

Prints classification report at the end of training

Uses weighted sampling to address class imbalance

Technologies Used

Python

PyTorch

Torchvision

NumPy

scikit-learn
