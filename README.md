# Diabetes Readmission Prediction Project


This project focuses on predicting hospital readmissions within 30 days for diabetic patients using the XGBoost machine learning algorithm. The dataset used originates from a large collection of hospital records, which undergoes thorough preprocessing, feature engineering, and model evaluation to build a robust classifier.

## Setup and Requirements
Install Python 3.7+
Required Libraries (install via pip if needed):
pip install pandas numpy scikit-learn seaborn matplotlib xgboost imbalanced-learn joblib


## Data Preprocessing Steps
### 1.Initial Data Cleaning
Removed duplicates and handled missing values (race, payer_code, diag_1, etc.)
Converted certain string-based columns into consistent labels (e.g., age bin mapping)
Dropped irrelevant or redundant features (e.g., encounter_id, patient_nbr)

### 2.Feature Engineering
Created new binary target variables:
readmitted_30: 1 if patient readmitted within 30 days
readmitted_any: 1 if patient was readmitted at all
Encoded medications and other categorical features
Grouped diagnosis codes using ICD-9 groupings
Created new features like med_change, num_diagnoses, weight_recorded

### 3. Feature Scaling
Applied StandardScaler to numerical features

# Model Training Pipeline
## Data Splitting
70% training / 30% testing split using train_test_split
SMOTE used to balance the imbalanced target classes

## Model: XGBoost
Hyperparameter tuning via RandomizedSearchCV
Parameters tested: n_estimators, max_depth, learning_rate, subsample

## Final Model
Trained with the best parameters from search
Saved as best_xgboost_model.pkl for deployment

# Model Evaluation
## 1. Metrics Reported
Accuracy, Precision, Recall, F1-Score
ROC AUC for binary classification performance


## 2.Visual Analysis
Confusion Matrix

ROC Curve

Precision-Recall Curve

Feature Importance (Top 20)

## 3. Classification Report
Detailed output including precision, recall, f1-score for each class (Readmit, No Readmit).

# Key Insights
SMOTE significantly improved model’s sensitivity to the minority class.

Certain features like num_inpatient, number_diagnoses, time_in_hospital, and specific medications had high predictive importance.

AUC and PRC curves provide robust visual validation of model performance.


# Outputs for Reporting
confusion_matrix.png

roc_curve.png

precision_recall_curve.png

feature_importance.png

Saved test set: diabetic_data_cleaned.csv

# Future Work
Integrate model into a web application for deployment

Explore more advanced models (e.g., LightGBM, Neural Networks)

Implement cross-site validation using patient demographic segmentation



