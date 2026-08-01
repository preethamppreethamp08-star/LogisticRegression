Loan Approval Prediction with Logistic Regression
Overview

This project uses logistic regression to predict loan approval outcomes based on applicants' financial information. It was built as a hands-on lab to practice the end-to-end machine learning workflow: data preprocessing, model training, and evaluation using classification metrics — applied to a real-world binary classification problem in the financial domain.

Dataset
500 loan applicants, 13 features including age, gender, education, income, employment experience, home ownership, loan amount, interest rate, credit history length, credit score, and prior default history.
Target variable: loan_status (1 = approved, 0 = not approved).
No missing values in the raw data.
Tools Used

Python, Pandas, Scikit-learn, Seaborn, Matplotlib, Jupyter Notebook

Process
Exploratory Data Analysis: Reviewed data structure, summary statistics, and null value counts.
Feature Selection: Selected three features for the initial model — person_income, credit_score, and loan_amnt.
Preprocessing: Split data into training/test sets (80/20, random_state=42) and applied feature scaling (standardization) to the training and test sets.
Model Training: Trained a Logistic Regression classifier on the scaled features.
Evaluation: Assessed performance using accuracy, precision, recall, a confusion matrix, and an ROC/AUC curve.
Results
Metric	Score
Accuracy	0.66
Precision	0.40
Recall	0.06
AUC	0.51
Limitations & Next Steps

This first iteration has clear room for improvement, and documenting that honestly is part of the exercise:

Low recall (0.06): The model currently misses most true loan approvals. This is likely driven by using only 3 of the 13 available features, and by class imbalance in loan_status (~20% approved) that wasn't explicitly addressed (e.g., via class weighting or resampling).
AUC near 0.51: This suggests the model performs close to random on ranking positive vs. negative cases, likely because the ROC/AUC step was computed on unscaled test data rather than the scaled data used for training — a preprocessing inconsistency worth fixing in the next version.
Planned improvements: incorporate more features (e.g., loan_intent, previous_loan_defaults_on_file, loan_percent_income), address class imbalance, and ensure consistent preprocessing across all evaluation steps.
Files
LogisticReg.ipynb — full notebook with code and outputs
loan_data.csv — dataset
