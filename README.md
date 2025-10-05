Team: TM19
Members: Leong Jia En Jesslyn,
Tan Bang Kai,
Chong Yuxuan

# CTG Triage — Datathon Project

This project builds an interpretable, reliable, and calibrated machine learning pipeline to classify **fetal cardiotocography (CTG)** signals into three clinical categories:
- **Normal**
- **Suspect**
- **Pathologic**

The dataset originates from the **UCI CTG Dataset**, containing ~2,000 samples with 21–35 signal summary features.

---

## 🧩 Project Overview

### Objectives
1. Predict fetal health condition (NSP) using CTG signal features.  
2. Build a trustworthy pipeline with proper cross-validation and calibration.  
3. Provide medically interpretable insights via feature importance and explainability.  
4. Demonstrate model robustness and reliability for clinical flagging use.

---

## ⚙️ Pipeline Summary

| Stage | Description |
|--------|--------------|
| **1. Data Cleaning** | Removed duplicates, converted non-numeric columns, filled missing with medians, standardized column names. |
| **2. Feature Engineering** | Added medically interpretable features (tachycardia, bradycardia, variability ratio, deceleration ratios, etc.). |
| **3. Model Ladder** | Dummy → Logistic Regression → SVM → Random Forest (with CV macro-F1 and balanced accuracy). |
| **4. Calibration & Evaluation** | Used 5-fold stratified CV, produced confusion matrix, ROC, PR, ECE plots. |
| **5. Interpretability** | Computed permutation importance for top features, saved plots and CSV. |
| **6. Robustness** | Simulated 1% Gaussian noise on numeric columns and computed Δ macro-F1 stability. |

---

## 📈 Key Outputs

All saved in `reports/figures/`:

- `confusion_matrix_oof.png` — aggregated 5-fold confusion matrix  
- `roc_ovr.png`, `pr_ovr.png` — ROC and PR curves (One-vs-Rest)  
- `calibration_ece_oof.csv` — per-class and macro Expected Calibration Error  
- `permutation_importance_top20_original.png` — top 20 features  
- `model_comparison.csv` — summary of all baseline models  
- `robustness.txt` — Δ macro-F1 from 1% numeric noise test  

---

## 🧠 Feature Definitions

| Feature | Description |
|----------|-------------|
| `feat_tachycardia` | Baseline FHR > 160 bpm |
| `feat_bradycardia` | Baseline FHR < 110 bpm |
| `feat_variability_ratio` | ALTV / (MSTV + ε) |
| `feat_accel_decel_balance` | (AC + 1) / (DL + DS + DP + 1) |
| `feat_decels_per_contraction` | (DL + DS + DP) / (UC + 1) |
| `feat_range` | MAX − MIN |
| `feat_instability_proxy` | WIDTH / (MSTV + 1) |

---

## 🧪 Running the Notebook

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Jupyter
jupyter lab
# Open notebooks/02_modeling.ipynb and run top to bottom
