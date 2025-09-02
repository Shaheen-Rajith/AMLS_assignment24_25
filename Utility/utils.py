import medmnist
from medmnist import INFO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import TensorDataset, DataLoader

def Download_Datasets():
    """
    This function downloads the BreastMNIST and BloodMNIST datasets
    and saves them as mentioned in the Assignment Handout in the following directory:
    Datasets/BreastMNIST/breastmnist.npz
    Datasets/BloodMNIST/bloodmnist.npz
    """
    datasets = {
        'BreastMNIST': 'breastmnist',
        'BloodMNIST': 'bloodmnist'
        }
    root = 'Datasets'
    os.makedirs(root, exist_ok = True)

    for f_name , d_name in datasets.items():
        dir = os.path.join(root, f_name)
        os.makedirs(dir, exist_ok = True)
        print(f'Downloading {f_name} Dataset')
        info = INFO[d_name]
        DatasetClass = getattr(medmnist, info['python_class'])
        train = DatasetClass(root = dir, split = 'train', download = True)
        val = DatasetClass(root = dir, split = 'val', download = True)
        test = DatasetClass(root = dir, split = 'test', download = True)
        print(f'Download Complete')


def PreProcess_SK(task):
    """
    This function uses the previously downloaded npz Dataset files, and does
    needed Data Preprocessing steps for Sci-Kit learn models like decision trees,
    bagged trees, the steps include normalization (0-1) and flattening (28x28 -> 784)
    The function returns the preprocessed train, val and test sets
    """
    if task.upper() == 'A':
        path = os.path.join('Datasets','BreastMNIST','breastmnist.npz')
    elif task.upper() == 'B':
        path = os.path.join('Datasets','BloodMNIST','bloodmnist.npz')
    else:
        return
    
    data = np.load(path)

    X_train = data["train_images"].astype(np.float32) / 255.0
    X_val = data["val_images"].astype(np.float32) / 255.0
    X_test = data["test_images"].astype(np.float32) / 255.0
    
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    y_train = data["train_labels"].squeeze().astype(np.int64)
    y_val = data["val_labels"].squeeze().astype(np.int64)
    y_test = data["test_labels"].squeeze().astype(np.int64)

    return X_train, X_val, X_test, y_train, y_val, y_test


def PreProcess_Torch(task, batch_size = 32):
    """
    This function uses the previously downloading npz Dataset files, and does
    needed Data Preprocessing steps for Pytorch DL models like CNNs, the steps 
    include normalization (0-1), ensuring there is a channel dimension, ensuring 
    that the order is channel first (N, C, H, W) as needed for Pytorch and 
    finally creating the DataLoader objects for train, val and test set.
    the function returns PyTorch DataLoader objects for train, val, test set
    """
    if task.upper() == 'A':
        path = os.path.join('Datasets','BreastMNIST','breastmnist.npz')
    elif task.upper() == 'B':
        path = os.path.join('Datasets','BloodMNIST','bloodmnist.npz')
    else:
        return
    
    data = np.load(path)

    X_train = data["train_images"].astype(np.float32) / 255.0
    X_val = data["val_images"].astype(np.float32) / 255.0
    X_test = data["test_images"].astype(np.float32) / 255.0

    if X_train.ndim == 3: 
        X_train = X_train[:, :, :, np.newaxis]
    if X_val.ndim == 3: 
        X_val = X_val[:, :, :, np.newaxis]
    if X_test.ndim == 3: 
        X_test = X_test[:, :, :, np.newaxis]
    
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_val = np.transpose(X_val, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    y_train = data["train_labels"].squeeze().astype(np.int64)
    y_val = data["val_labels"].squeeze().astype(np.int64)
    y_test = data["test_labels"].squeeze().astype(np.int64)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds  = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    val_loader   = DataLoader(val_ds,   batch_size = batch_size, shuffle = False)
    test_loader  = DataLoader(test_ds,  batch_size = batch_size, shuffle = False)

    return train_loader, val_loader, test_loader