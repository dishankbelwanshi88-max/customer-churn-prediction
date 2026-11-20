# Project Report: Telecom Customer Churn Prediction

**Author:** Dishank Belwanshi  
**Project category:** Data Science / Machine Learning  
**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit

## 1. Executive summary
This project develops an end-to-end workflow to explore customer churn in a telecom sample dataset and compare two supervised classification approaches. The intended business use is to help a retention team prioritize customers for further review.

## 2. Problem statement
A telecom provider may lose recurring revenue when customers discontinue a service. The project asks: can customer account, contract and service attributes help estimate which customers are more likely to churn?

## 3. Dataset
The project uses the IBM Telco Customer Churn sample dataset. It contains 7,043 customer records and a binary `Churn` target. The dataset includes demographic, account, billing and subscribed-service attributes. The original `customerID` is excluded from model training because it is an identifier rather than a meaningful predictive feature.

Source: IBM public sample-data repository (linked in README).

## 4. Data preparation
- Convert `TotalCharges` from text to numeric.
- Treat invalid/blank numeric entries as missing and impute within the training pipeline.
- Map `Churn` from Yes/No to 1/0.
- Separate predictors from the target.
- Use a stratified 80/20 train/test split with a fixed random seed.
- Apply median imputation and standard scaling to numeric columns.
- Apply most-frequent imputation and one-hot encoding to categorical columns.

## 5. Exploratory analysis
The dashboard and notebook inspect:
- Target-class distribution.
- Churn patterns by contract type.
- Average monthly charges by churn status.
- Dataset shape, data types and missing values.

## 6. Modelling
Two models are compared:
1. Logistic Regression with balanced class weights — an interpretable baseline.
2. Random Forest with balanced class weights — a non-linear tree ensemble.

Preprocessing and modelling are combined in Scikit-learn Pipelines so transformations are learned from training data only.

## 7. Evaluation
The training script calculates accuracy, precision, recall, F1-score and ROC-AUC on the held-out test set. It also saves ROC and confusion-matrix plots.

**Insert the actual results after running the project.** The project intentionally does not state invented performance scores. Copy the values from `outputs/model_metrics.csv` into this table.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | Run project | Run project | Run project | Run project | Run project |
| Random Forest | Run project | Run project | Run project | Run project | Run project |

Recall is important in a churn-retention setting because missed churners may represent missed opportunities for outreach. Precision also matters because unnecessary outreach can consume resources. The preferred model should be selected with the business trade-off in mind, not accuracy alone.

## 8. Deliverables
- Reproducible data download script.
- Exploratory analysis notebook.
- Model training and evaluation pipeline.
- Streamlit dashboard with descriptive analysis and an example prediction form.
- Metrics and visualizations generated at runtime.

## 9. Limitations and responsible use
The dataset is a sample and may not represent current customers or current market conditions. Predictive associations are not causal explanations. Model probabilities are not certainties. Any real-world deployment should include validation on recent representative data, probability calibration, fairness checks, privacy controls and ongoing monitoring.

## 10. Conclusion
The project demonstrates a complete applied Data Science workflow: data acquisition, cleaning, exploratory analysis, preprocessing, supervised learning, evaluation and an interactive application. The next step is to run the project, inspect the generated metrics and discuss the model's strengths and limitations.
