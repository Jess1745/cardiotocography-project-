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

- [Insert your model’s accuracy, F1-score, or confusion matrix screenshot here]

## 📓 Notebook

- See [`ctg_analysis.ipynb`](ctg_analysis.ipynb) for full code and outputs.

## ✅ Next Steps

- Try other models (e.g., SVM, XGBoost)
- Tune hyperparameters
- Handle class imbalance (e.g., with SMOTE)
