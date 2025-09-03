import medmnist
from medmnist import INFO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm



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
    This function uses the previously downloaded npz Dataset files, and does
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


def Random_Sample_Visual(x,y,loader,name):
    '''
    
    '''
    fig , axs = plt.subplots(x,y)
    images, labels = next(iter(loader))
    images, labels = next(iter(loader))
    for i in range(x):
        for j in range(y):
            with torch.no_grad():
                index = torch.randint(len(images) , size=(1,)).item()
                img = images[index]
                label = labels[index]
                if img.shape[0] == 1:
                    axs[i][j].imshow(img.permute(1, 2, 0), cmap="gray")
                else:
                    axs[i][j].imshow((img).squeeze().permute(1, 2, 0), cmap='binary')
                axs[i][j].set_title(f'Class {label}')
                axs[i][j].set_xticks([])
                axs[i][j].set_yticks([])
    plt.suptitle('Random Samples from Training Data with Class')
    fig.tight_layout()
    dir = os.path.join('Results',name)
    plt.savefig(dir)

def TestModel(true_labels, predicted_labels, label_names):
    """
    This function evaluates the ML models and gives performance metric scores for accuracy,
    precision, recall, F1 score and generates the classification report as well as the confusion 
    matrix. The function takes y_true (from the test set) and y_pred (model predictions for X_test)
    along with the class names as parameters and returns the confusion matrix.
    """
    # Calculates accuracry, precision, recall and f1 scores.
    print(f'Accuracy: {accuracy_score(true_labels, predicted_labels)}')
    print(f'Precision: {precision_score(true_labels, predicted_labels, average="weighted")}')
    print(f'Recall: {recall_score(true_labels, predicted_labels, average="weighted")}')
    print(f'F1 Score: {f1_score(true_labels, predicted_labels, average="weighted")}')

    # Generate Classification Report
    print('Classification Report: ')
    print(classification_report(true_labels, predicted_labels, target_names=label_names))

    # Generates confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    return cm



def CM_Display(cm, class_names,title):
    num = len(class_names)
    fig , ax = plt.subplots()
    ax.imshow(cm,cmap='binary')
    ax.set_xticks(np.arange(0, num), labels=class_names,  rotation=90)
    ax.set_yticks(np.arange(0, num), labels=class_names)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('Actual Test Labels')
    plt.suptitle('Confusion Matrix')
    for i in range(num):
        for j in range(num):
            ax.text(i, j, cm[j, i], ha="center", va="center", color="r")
    save_path = "Results/"+title+".png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Confusion Matrix obtained and saved at: {save_path}")

def CNN_Train(model, train_loader, val_loader, n_epochs, lr, device, criterion):
    train_losses = []
    val_losses = []
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        # Training
        model.train()
        running_loss, total = 0.0, 0

        # tqdm Progess Bar setup
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            #forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            #backprop
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            total += images.size(0)
        avg_train_loss = running_loss / total
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                #only forward pass since no training
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_total += images.size(0)
            avg_val_loss = val_loss / val_total
            val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    return model, train_losses, val_losses



def TrainingPlots(train_losses,val_losses,n_epochs,title):
    plt.figure(figsize=(10,6))
    plt.plot(range(1, n_epochs+1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, n_epochs+1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training And Validation Losses over Epochs')
    plt.legend()
    plt.grid(True)
    save_path = "Results/"+title+".png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Training Loss plot obtained and saved at: {save_path}")

def TestModel_Torch(model, data_loader, device,class_names,task):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    cm = TestModel(targets,preds,class_names)
    if (task == 'B'):
        CM_Display(cm, class_names,'CM_B_CNN')
    else:
        CM_Display(cm, class_names,'CM_A_CNN')