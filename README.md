# cardiotocography-project-
Interpret patterns hidden in the baby's heart rate  
# Cardiotocography (CTG) Classification

This project analyzes and models fetal health using the CTG dataset from the UCI Machine Learning Repository.

## 📁 Dataset

- Source: [UCI CTG Dataset](https://archive.ics.uci.edu/dataset/193/cardiotocography)
- Description: Contains fetal cardiotocograms measured during pregnancy, classified into 3 fetal states:
  - 1 = Normal
  - 2 = Suspect
  - 3 = Pathologic

## 🧹 Preprocessing

- Loaded clean sheet from `CTG.xls`
- Dropped rows with missing `NSP` (target variable)
- Selected only numeric columns
- Visualized feature correlations

## 🔍 Exploratory Data Analysis

- Correlation heatmap of features
- Target class distribution

## 🤖 Model Development

- Random Forest Classifier
- Train-test split (80/20, stratified)
- Evaluation using:
  - Classification report (precision, recall, F1-score)
  - Confusion matrix

## 📊 Results

Classification Report:
              precision    recall  f1-score   support

         1.0       0.98      1.00      0.99       332
         2.0       0.98      0.86      0.92        59
         3.0       1.00      1.00      1.00        35

    accuracy                           0.98       426
   macro avg       0.99      0.95      0.97       426
weighted avg       0.98      0.98      0.98       426


## 📓 Notebook

- See [`ctg_analysis.ipynb`](ctg_exploration.ipynb) for full code and outputs.

