# Telecom Customer Churn Prediction

**End-to-end Data Science project | Python • EDA • Machine Learning • Streamlit**

## Project overview
This project analyzes the IBM Telco Customer Churn sample dataset and builds machine-learning models to estimate whether a customer is likely to leave a telecom provider. The business goal is to help a retention team identify customers who may need proactive support.

This is an educational portfolio project. Model predictions are estimates, not certainty, and should not be used as the sole basis for customer decisions.

## Problem statement
Customer churn can reduce recurring revenue and increase acquisition costs. The project explores customer/account/service attributes, identifies patterns associated with churn, compares classification models, and provides an interactive prediction interface.

## Dataset
- **Dataset:** IBM Telco Customer Churn sample dataset
- **Size:** 7,043 customer records; 21 columns (including customer ID and target)
- **Target:** `Churn` (`Yes` / `No`)
- **Source:** IBM public sample-data repository: https://github.com/IBM/telco-customer-churn-on-icp4d
- **Direct CSV:** https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- The CSV is downloaded by the data loader when you run the project; it is not duplicated in this ZIP.

## Workflow
1. Download and inspect the dataset.
2. Clean `TotalCharges` and handle missing values.
3. Perform exploratory data analysis (EDA).
4. Split data into train/test sets using stratification.
5. Build preprocessing pipelines for numeric and categorical features.
6. Compare Logistic Regression and Random Forest classifiers.
7. Evaluate with accuracy, precision, recall, F1-score and ROC-AUC.
8. Save evaluation outputs and a trained model.
9. Explore the data and score a sample customer in the Streamlit app.

## Repository structure
```text
customer-churn-prediction/
├── app.py
├── requirements.txt
├── README.md
├── REPORT.md
├── .gitignore
├── data/
│   └── raw/                 # downloaded CSV is stored here
├── notebooks/
│   └── 01_customer_churn_analysis.ipynb
├── outputs/                 # generated charts and metrics
└── src/
    ├── data_loader.py
    └── train.py
```

## Setup
Python 3.10 or newer is recommended.

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv .venv
```

Activate the environment:
- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install packages:
```bash
pip install -r requirements.txt
```

Download the dataset:
```bash
python -m src.data_loader
```

Train and evaluate:
```bash
python -m src.train
```

Launch the dashboard:
```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).

## Expected outputs
After running training, the `outputs/` folder will contain:
- `model_metrics.csv` — held-out test metrics for each model
- `confusion_matrix.png` — confusion matrix for the selected model
- `roc_curve.png` — ROC curve comparison
- `feature_importance.png` — permutation feature importance
- `best_model.joblib` — fitted model pipeline

**Important:** Results are generated on your machine when you run the code. Do not claim performance numbers until you have run the pipeline and checked `outputs/model_metrics.csv`.

## Evaluation notes
Churn datasets are often imbalanced. Accuracy alone can be misleading, so recall, precision, F1-score and ROC-AUC are reported together. The train/test split is stratified, and preprocessing is fitted only on training data through an sklearn Pipeline to reduce data leakage.

## Limitations
- The dataset is a sample dataset and may not represent a current telecom customer population.
- The model learns associations, not causal relationships.
- Performance may change with a different split, threshold, or production data.
- Before real deployment, validate on recent data, calibrate probabilities, assess fairness, and monitor model drift.

## Author
**Dishank Belwanshi**  
GitHub: https://github.com/dishankbelwanshi88-max

## Suggested project walkthrough
1. Explain the business problem and target variable.
2. Show the data cleaning and EDA notebook.
3. Explain why a stratified split and preprocessing pipeline are used.
4. Compare Logistic Regression with Random Forest using recall and ROC-AUC, not accuracy alone.
5. Demonstrate a sample prediction in Streamlit.
6. Discuss limitations and responsible use.
