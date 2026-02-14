# Project-Based-Learning


## 1. Project Overview
This project focuses on predicting genetic risk associated with prenatal screening using machine learning models.  
The system analyzes genomic variant data and classifies variants to support early detection of potential chromosomal abnormalities.

The primary objective is to build a robust predictive model that can assist in decision-making for prenatal genetic screening.

---

## 2. Problem Statement
Traditional prenatal screening methods may generate false positives, leading to unnecessary invasive procedures such as amniocentesis.  
This project aims to develop a machine learning-based classification system to improve prediction accuracy and reduce unnecessary medical interventions.

---

## 3. Dataset
- Source: gnomAD genomic variant dataset
- Processed file: `gnomad_labeled.csv`
- Gene-specific variant files (e.g., BRCA1, HIST genes)
- Labeled data used for supervised learning

---

## 4. Project Workflow

1. Data Collection  
2. Data Cleaning (`cleaned_data.py`)  
3. Feature Engineering  
4. Train-Test Split  
5. Model Training  
   - XGBoost (Primary Model)  
   - Random Forest  
   - Transformer  
6. Model Evaluation  
7. Model Saving (`saved_models/`)  

---

## 5. Models Used

### XGBoost
- Primary classification model
- Handles structured genomic data efficiently
- Includes regularization to prevent overfitting
- Provides feature importance analysis

### Random Forest
- Baseline ensemble model
- Used for comparative performance evaluation

### Transformer
- Sequence-based model for learning deeper patterns in genomic data

---

## 6. Evaluation Metrics
- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- Confusion Matrix

---

## 7. Project Structure
```
Project-Based-Learning/
│
├── datasets/ # Raw and processed genomic datasets
│
├── logs/ # Training and experiment logs
│ └── lightning_logs/ # PyTorch Lightning training logs
│ └── readme.md # Description of logging details
│
├── models/ # Trained and exported models
│ ├── saved_models/ # Serialized model files (.pkl, .pt, etc.)
│ └── readme.md # Model storage documentation
│
├── src/ # Source code for data processing and models
│ ├── XGboost.py # XGBoost training and evaluation script
│ ├── random_forest.py # Random Forest model implementation
│ ├── transformer.py # Transformer-based model implementation
│ ├── cleaned_data.py # Data cleaning and preprocessing pipeline
│ ├── data.py # Data loading and dataset handling utilities
│ └── readme.md # Source code documentation
│
└── README.md # Main project documentation
```
---

## 8. Expected Outcome
- Accurate classification of genomic variants  
- Comparative analysis of multiple ML models  
- Optimized predictive model for prenatal genetic screening support  

---

## 9. Future Improvements
- Integration with real clinical prenatal datasets  
- Hyperparameter optimization  
- Model explainability using SHAP or similar techniques  
- Deployment as a web-based decision support tool  

---

## 10. Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- torch (for Transformer model)

