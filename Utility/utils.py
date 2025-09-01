import medmnist
from medmnist import INFO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

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


def PreProcess_SK(path):
    """
    
    """
