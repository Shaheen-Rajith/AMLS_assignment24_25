import os
import torch
import torch.nn as nn
import gc
import time

from Utility.utils import Download_Datasets, PreProcess_SK, PreProcess_Torch
from Utility.utils import TestModel_Torch, Random_Sample_Visual, TrainingPlots
from A.A_DecisionTree import A_DT_Train
from A.A_BaggedDT import A_BaggingDT
from A.A_CNN import A_CNN, A_CNN_Train

#from B.B_DecisionTree import B_DT_Train
#from B.B_BaggedDT import B_BaggingDT
#from B.B_CNN import B_CNN, B_CNN_Train

# ======================================================================================================================
# Dataset Downloading
Download_Datasets()

# ======================================================================================================================
# Task A - Preprocessing
print('============================== Task - A (BloodMNIST)  ==============================')
X_train, X_val, X_test, y_train, y_val, y_test = PreProcess_SK('A')
train_loader, val_loader, test_loader = PreProcess_Torch('A', batch_size = 32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Beneign', 'Malignant']

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

model_A_CNN, train_losses, val_losses = A_CNN_Train(
    model = model_A_CNN,
    train_loader = train_loader,
    val_loader = val_loader,
    n_epochs = n_epochs,
    lr = 5e-5,
    device = device,
    criterion = nn.CrossEntropyLoss()
)

TrainingPlots(train_losses,val_losses,n_epochs,'A_CNN_loss')
TestModel_Torch(model_A_CNN,test_loader,device,class_names)
time.sleep (4)

gc.collect()
torch.cuda.empty_cache()





# # ======================================================================================================================
# ## Print out your results with following format:
# print('TA (Custom CNN):{:.5f},{:.5f}; TA (Transfer Learning):{:.5f},{:.5f};'.format(acc_A_train_self, acc_A_test_self,
#                                                         acc_A_train_trans, acc_A_test_trans))

# # If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# # acc_A_train = 'TBD'
# # acc_B_test = 'TBD'