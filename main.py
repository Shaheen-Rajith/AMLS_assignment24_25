import os
import torch
import torch.nn as nn
import gc
import time

from Utility.utils import Download_Datasets, PreProcess_SK, PreProcess_Torch, Random_Sample_Visual
from Utility.utils import CNN_Train, TestModel_Torch, TrainingPlots

from A.A_DecisionTree import A_DT_Train
from A.A_BaggedDT import A_BaggingDT
from A.A_CNN import A_CNN

from B.B_DecisionTree import B_DT_Train
from B.B_BaggedDT import B_BaggingDT
from B.B_CNN import B_CNN 

# ======================================================================================================================
# Dataset Downloading
Download_Datasets()

# ======================================================================================================================
# Task A
print('\n\n\n\n\n\n\n\n\n\n============================== Task - A (BreastMNIST)  ==============================')
# Preprocessing dataset and converting into needed format for both pytorch CNN and SK-Learn models 
X_train, X_val, X_test, y_train, y_val, y_test = PreProcess_SK('A')
train_loader, val_loader, test_loader = PreProcess_Torch('A', batch_size = 32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Benign', 'Malignant']

Random_Sample_Visual(2,2,train_loader,'A')

# ======================================================================================================================
# Model 1 - Decision Tree
print('\n\n\n\n--------- Model 1 - Decision Tree   ---------')
A_DT_Train(X_train, X_val, X_test, y_train, y_val, y_test,class_names)
time.sleep (4)

# Model 2 - Bagging with Decision Trees
print('\n\n\n\n--------- Model 2 - Bagging with Decision Trees   ---------')
A_BaggingDT(X_train, X_val, X_test, y_train, y_val, y_test,class_names)
time.sleep (4)

# Model 3 - Pytorch CNN
print('\n\n\n\n--------- Model 3 - Pytorch CNN   ---------')
model_A_CNN = A_CNN(num_classes= 2)  # Build model object.
n_epochs = 30

# Training CNN
model_A_CNN, train_losses, val_losses = CNN_Train(
    model = model_A_CNN,
    train_loader = train_loader,
    val_loader = val_loader,
    n_epochs = n_epochs,
    lr = 5e-5,
    device = device,
    criterion = nn.CrossEntropyLoss()
)

# Obtaining loss curves plot
TrainingPlots(train_losses,val_losses,n_epochs,'A_CNN_loss')
# Checking model performance on Test set
TestModel_Torch(model_A_CNN,test_loader,device,class_names,'A')
time.sleep (4)

gc.collect()
torch.cuda.empty_cache()

# ======================================================================================================================
# Task B 
print('\n\n\n\n\n\n\n\n\n\n============================== Task - B (BloodMNIST)  ==============================')
# Preprocessing dataset and converting into needed format for both pytorch CNN and SK-Learn models 
X_train, X_val, X_test, y_train, y_val, y_test = PreProcess_SK('B')
train_loader, val_loader, test_loader = PreProcess_Torch('B', batch_size = 32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['basophil', 'eosinophil', 'erythroblast', 'immature granulocytes', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

Random_Sample_Visual(2,2,train_loader,'B')

# ======================================================================================================================
# Model 1 - Decision Tree
print('\n\n\n\n--------- Model 1 - Decision Tree   ---------')
B_DT_Train(X_train, X_val, X_test, y_train, y_val, y_test,class_names)
time.sleep (4)

# Model 2 - Bagging with Decision Trees
print('\n\n\n\n--------- Model 2 - Bagging with Decision Trees   ---------')
B_BaggingDT(X_train, X_val, X_test, y_train, y_val, y_test,class_names)
time.sleep (4)

# Model 3 - Pytorch CNN
print('\n\n\n\n--------- Model 3 - Pytorch CNN   ---------')
model_B_CNN = B_CNN(num_classes= 8)  # Build model object.
n_epochs = 25

# Training CNN
model_B_CNN, train_losses, val_losses = CNN_Train(
    model = model_B_CNN,
    train_loader = train_loader,
    val_loader = val_loader,
    n_epochs = n_epochs,
    lr = 3e-5,
    device = device,
    criterion = nn.CrossEntropyLoss()
)

# Obtaining loss curves plot
TrainingPlots(train_losses,val_losses,n_epochs,'B_CNN_loss')
# Checking model performance on Test set
TestModel_Torch(model_B_CNN,test_loader,device,class_names,'B')
time.sleep (4)