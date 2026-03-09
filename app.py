import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="HW1 - Big4 AI Auditing", layout="wide")

# Load data
df = pd.read_csv("data/big4_financial_risk_compliance.csv")
results_df = pd.read_csv("models/model_results.csv")

# Load models
log_model = joblib.load("models/logistic_model.joblib")
tree_model = joblib.load("models/decision_tree_model.joblib")
rf_model = joblib.load("models/random_forest_model.joblib")
xgb_model = joblib.load("models/xgb_model.joblib")

target = "AI_Used_for_Auditing"

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Interactive Prediction"
])

# -----------------------------
# Tab 1: Executive Summary
# -----------------------------
with tab1:
    st.title("Executive Summary")

    st.write("""
    This project predicts whether AI was used for auditing in a Big Four financial risk and compliance dataset.
    The dataset includes engagement-level operational, risk, fraud, workload, industry, and quality-related variables.
    The target variable is `AI_Used_for_Auditing`, which indicates whether the case involved AI-enabled auditing.
    """)

    st.write("""
    This prediction problem is meaningful because AI adoption in audit settings affects efficiency, staffing,
    consistency, fraud detection, and audit transformation strategy. Understanding what conditions are associated
    with AI-enabled auditing can help firms allocate technology resources more effectively.
    """)

    st.write("""
    The workflow includes descriptive analytics, preprocessing, five predictive models, SHAP-based explainability,
    and an interactive Streamlit deployment. The deployed app allows users to review results and generate new predictions.
    """)

# -----------------------------
# Tab 2: Descriptive Analytics
# -----------------------------
with tab2:
    st.title("Descriptive Analytics")

    st.image(
        "outputs/target_distribution.png",
        caption="The target distribution is relatively balanced, which supports stable classification modeling."
    )

    st.image(
        "outputs/viz_1.png",
        caption="This figure compares audit effectiveness across AI and non-AI cases."
    )

    st.image(
        "outputs/viz_2.png",
        caption="This figure shows how employee workload differs between AI and non-AI auditing cases."
    )

    st.image(
        "outputs/viz_3.png",
        caption="This chart compares AI adoption rates across industries."
    )

    st.image(
        "outputs/viz_4.png",
        caption="This plot shows the relationship between high-risk cases and fraud cases detected, colored by AI usage."
    )

    st.image(
        "outputs/correlation_heatmap.png",
        caption="The heatmap summarizes linear relationships among numerical variables."
    )

# -----------------------------
# Tab 3: Model Performance
# -----------------------------
with tab3:
    st.title("Model Performance")

    st.subheader("Model Comparison Table")
    st.dataframe(results_df, use_container_width=True)

    st.image(
        "outputs/model_comparison.png",
        caption="This bar chart compares all models using F1 score."
    )

    st.image("outputs/roc_logistic.png", caption="ROC curve for Logistic Regression.")
    st.image("outputs/roc_tree.png", caption="ROC curve for Decision Tree.")
    st.image("outputs/roc_rf.png", caption="ROC curve for Random Forest.")
    st.image("outputs/roc_xgb.png", caption="ROC curve for XGBoost.")
    st.image("outputs/mlp_history.png", caption="Training loss curve for the PyTorch MLP.")

    st.subheader("Best Hyperparameters")
    st.write("Please paste your final best hyperparameters here after training in Colab.")

# -----------------------------
# Tab 4: Explainability & Interactive Prediction
# -----------------------------
with tab4:
    st.title("Explainability & Interactive Prediction")

    st.subheader("SHAP Explainability")
    st.image("outputs/shap_summary.png", caption="SHAP summary plot.")
    st.image("outputs/shap_bar.png", caption="SHAP bar plot.")
    st.image("outputs/shap_waterfall.png", caption="SHAP waterfall plot for one sample prediction.")

    st.subheader("Interactive Prediction")

    model_choice = st.selectbox(
        "Select a model",
        ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
    )

    selected_model = {
        "Logistic Regression": log_model,
        "Decision Tree": tree_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model
    }[model_choice]

    year = st.slider("Year", int(df["Year"].min()), int(df["Year"].max()), int(df["Year"].median()))
    total_eng = st.slider("Total Audit Engagements", int(df["Total_Audit_Engagements"].min()), int(df["Total_Audit_Engagements"].max()), int(df["Total_Audit_Engagements"].median()))
    high_risk = st.slider("High Risk Cases", int(df["High_Risk_Cases"].min()), int(df["High_Risk_Cases"].max()), int(df["High_Risk_Cases"].median()))
    violations = st.slider("Compliance Violations", int(df["Compliance_Violations"].min()), int(df["Compliance_Violations"].max()), int(df["Compliance_Violations"].median()))
    fraud = st.slider("Fraud Cases Detected", int(df["Fraud_Cases_Detected"].min()), int(df["Fraud_Cases_Detected"].max()), int(df["Fraud_Cases_Detected"].median()))
    revenue = st.slider("Total Revenue Impact", float(df["Total_Revenue_Impact"].min()), float(df["Total_Revenue_Impact"].max()), float(df["Total_Revenue_Impact"].median()))
    workload = st.slider("Employee Workload", int(df["Employee_Workload"].min()), int(df["Employee_Workload"].max()), int(df["Employee_Workload"].median()))
    effectiveness = st.slider("Audit Effectiveness Score", float(df["Audit_Effectiveness_Score"].min()), float(df["Audit_Effectiveness_Score"].max()), float(df["Audit_Effectiveness_Score"].median()))
    satisfaction = st.slider("Client Satisfaction Score", float(df["Client_Satisfaction_Score"].min()), float(df["Client_Satisfaction_Score"].max()), float(df["Client_Satisfaction_Score"].median()))
    firm = st.selectbox("Firm Name", sorted(df["Firm_Name"].unique()))
    industry = st.selectbox("Industry Affected", sorted(df["Industry_Affected"].unique()))

    input_df = pd.DataFrame([{
        "Year": year,
        "Firm_Name": firm,
        "Total_Audit_Engagements": total_eng,
        "High_Risk_Cases": high_risk,
        "Compliance_Violations": violations,
        "Fraud_Cases_Detected": fraud,
        "Industry_Affected": industry,
        "Total_Revenue_Impact": revenue,
        "Employee_Workload": workload,
        "Audit_Effectiveness_Score": effectiveness,
        "Client_Satisfaction_Score": satisfaction
    }])

    pred_prob = selected_model.predict_proba(input_df)[0, 1]
    pred_class = "Yes" if pred_prob >= 0.5 else "No"

    st.write(f"### Predicted AI Usage: {pred_class}")
    st.write(f"### Predicted Probability of AI Usage: {pred_prob:.3f}")
