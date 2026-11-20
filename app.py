"""Streamlit dashboard for telecom churn exploration and prediction."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from src.data_loader import load_data
from src.train import prepare_data, make_pipeline
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "best_model.joblib"

st.set_page_config(page_title="Customer Churn Intelligence", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Intelligence")
st.caption("Exploratory analysis and machine-learning demonstration using the IBM Telco sample dataset.")

@st.cache_data
def get_data():
    return load_data()

try:
    df = get_data()
except Exception as exc:
    st.error(f"Could not load the dataset. Check your internet connection and try again. Details: {exc}")
    st.stop()

clean = df.copy()
clean["TotalCharges"] = pd.to_numeric(clean["TotalCharges"], errors="coerce")
clean = clean.dropna(subset=["Churn"])

c1, c2, c3 = st.columns(3)
c1.metric("Customer records", f"{len(clean):,}")
c2.metric("Churn rate", f"{(clean['Churn'].eq('Yes').mean()*100):.1f}%")
c3.metric("Average monthly charge", f"${clean['MonthlyCharges'].mean():.2f}")

tab1, tab2, tab3 = st.tabs(["Overview", "Churn patterns", "Predict a customer"])

with tab1:
    st.subheader("Dataset preview")
    st.dataframe(clean.head(15), use_container_width=True)
    st.subheader("Churn distribution")
    counts = clean["Churn"].value_counts().rename_axis("Customer status").to_frame("Customers")
    st.bar_chart(counts)

with tab2:
    left, right = st.columns(2)
    with left:
        st.subheader("Churn by contract")
        contract_churn = pd.crosstab(clean["Contract"], clean["Churn"], normalize="index") * 100
        st.bar_chart(contract_churn)
    with right:
        st.subheader("Monthly charges by churn status")
        charge_summary = clean.groupby("Churn")["MonthlyCharges"].mean()
        st.bar_chart(charge_summary)
    st.info("These are descriptive associations in this sample; they do not establish causation.")

with tab3:
    st.subheader("Customer churn-risk estimate")
    st.write("Enter an example customer's account and service details. The model must be trained first.")
    if not MODEL_PATH.exists():
        st.warning("No trained model found. Run `python -m src.train` from the project folder, then refresh this page.")
    else:
        model = joblib.load(MODEL_PATH)
        defaults = clean.drop(columns=["Churn", "customerID"], errors="ignore")
        sample = defaults.mode(dropna=True).iloc[0].to_dict()
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
        categorical_cols = [c for c in defaults.columns if c not in numeric_cols]
        with st.form("customer_form"):
            inputs = {}
            col_a, col_b = st.columns(2)
            for i, col in enumerate(numeric_cols):
                if col in defaults.columns:
                    with (col_a if i % 2 == 0 else col_b):
                        maximum = int(max(1, clean[col].max())) if col != "MonthlyCharges" and col != "TotalCharges" else float(max(1, clean[col].max()))
                        if col == "SeniorCitizen":
                            inputs[col] = st.selectbox(col, [0, 1], index=int(sample[col]))
                        elif col == "tenure":
                            inputs[col] = st.slider(col + " (months)", 0, 72, int(sample[col]))
                        else:
                            inputs[col] = st.number_input(col, min_value=0.0, max_value=float(maximum), value=float(sample[col]))
            for col in categorical_cols:
                options = sorted(defaults[col].dropna().astype(str).unique().tolist())
                if options:
                    inputs[col] = st.selectbox(col, options, index=options.index(str(sample[col])) if str(sample[col]) in options else 0)
            submitted = st.form_submit_button("Estimate churn risk")
        if submitted:
            row = pd.DataFrame([inputs], columns=defaults.columns)
            probability = float(model.predict_proba(row)[0, 1])
            st.metric("Estimated churn probability", f"{probability:.1%}")
            if probability >= 0.5:
                st.warning("The model classifies this example as higher churn risk (threshold: 50%). Consider reviewing the customer's needs before taking action.")
            else:
                st.success("The model classifies this example as lower churn risk (threshold: 50%).")
            st.caption("This is a model estimate, not a guarantee or a decision about an actual person.")
