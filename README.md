# ELEC0134 AMLS assignment 2024/2025 

This project consists of two task A & B, Task A is binary classification of the BreastMNIST dataset where images are labelled as either benign or malignant while Task B is multiclass classification of the BloodMNIST dataset where images are of blood cells belonging to one of 8 classes. Three models have been tested for both tasks, they are: Single Decision Tree, Ensemble of Bagged Decision Trees and CNN based model.

## Project Structure
 - main.py , Main File, running it will train and evaluate all 6 models corresponding to both tasks.

 - A/ , Source Code for Models related to Task A :
    + A_BaggedDT.py , model Code for ensemble of bagged decision trees model
    + A_CNN.py , model Code for CNN model
    + A_DecisionTree.py , model Code for single decision tree model

 - B/ , Source Code for Models related to Task B :
    + B_BaggedDT.py , model Code for ensemble of bagged decision trees model
    + B_CNN.py , model Code for CNN model
    + B_DecisionTree.py , model Code for single decision tree model
 
 - Utility/
    + utils.py , Contains code for downloading datasets, preprocessing data (for both tree based models and CNN models), training and testing of CNN model, plotting loss curves and confusion matrices, and generating random sample of training dataset.

- Results , Used for storing loss curves, confusion matrices and other results.
- env/
    + environment.yml , Code to create a new conda env called "AMLS" with all necessary modules
    + requirements.txt, All needed modules
- README.md , This file

## All Packages needed to run the code
- numpy
- matplotlib
- torch
- tqdm
- scikit-learn
- medmnist

## Instructions
git clone the repo, go into the project root folder and open terminal.
Run the following code to create a new conda environment named "AMLS" with all the necessary modules.
```bash
conda env create -f env/environment.yml
```
Go back to the project root folder and run the following code to train the all 3 models for both Task A & B, obtain loss plots, performance metrics and confusion matrices (takes around 10 - 15 mins to run on my laptop with Nvidia GPU)
```bash
python3 main.py
```