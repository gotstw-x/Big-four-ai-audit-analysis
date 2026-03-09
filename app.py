import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="HW1 - Big4 AI Auditing", layout="wide")

# -----------------------------
# Helper functions
# -----------------------------
def show_image_with_analysis(image_path, caption, analysis_text):
    if os.path.exists(image_path):
        st.image(image_path, caption=caption, use_container_width=True)
        st.write(analysis_text)
        st.markdown("---")
    else:
        st.warning(f"Missing file: {image_path}")

def safe_load_joblib(path):
    return joblib.load(path) if os.path.exists(path) else None

# -----------------------------
# Load data
# -----------------------------
if os.path.exists("data/big4_financial_risk_compliance.csv"):
    df = pd.read_csv("data/big4_financial_risk_compliance.csv")
else:
    st.error("The dataset file `data/big4_financial_risk_compliance.csv` is missing.")
    st.stop()

# Try reading saved results; if missing, use the notebook-confirmed values
if os.path.exists("models/model_results.csv"):
    results_df = pd.read_csv("models/model_results.csv")
else:
    results_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "MLP"],
        "Accuracy": [0.533333, 0.400000, 0.366667, 0.600000, 0.533333],
        "Precision": [0.454545, 0.307692, 0.250000, 0.538462, 0.461538],
        "Recall": [0.384615, 0.307692, 0.230769, 0.538462, 0.461538],
        "F1": [0.416667, 0.307692, 0.240000, 0.538462, 0.461538],
        "AUC_ROC": [0.497738, 0.457014, 0.466063, 0.647059, 0.461538]
    })

# Load models
log_model = safe_load_joblib("models/logistic_model.joblib")
tree_model = safe_load_joblib("models/decision_tree_model.joblib")
rf_model = safe_load_joblib("models/random_forest_model.joblib")
xgb_model = safe_load_joblib("models/xgb_model.joblib")

target = "AI_Used_for_Auditing"

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Interactive Prediction"
])

# =============================
# TAB 1: Executive Summary
# =============================
with tab1:
    st.title("Executive Summary")

    st.write("""
    This project analyzes a Big Four financial risk and compliance dataset and predicts whether AI was used for auditing.
    The target variable is `AI_Used_for_Auditing`, and the predictors include engagement scale, risk burden, compliance violations,
    fraud detection activity, industry context, revenue impact, employee workload, audit effectiveness, and client satisfaction.
    The dataset contains 100 rows and 12 columns, combining both numerical and categorical features.
    """)

    st.write("""
    This problem matters because AI adoption in auditing is directly related to efficiency, operational complexity, fraud monitoring,
    staffing pressure, and audit transformation. If firms can better understand the conditions associated with AI-enabled auditing,
    they can allocate technology investments more strategically and identify the types of engagements where AI is most valuable.
    """)

    st.write("""
    The workflow included descriptive analytics, supervised classification models, SHAP-based explainability, and Streamlit deployment.
    Five models were compared: Logistic Regression, Decision Tree, Random Forest, XGBoost, and a PyTorch MLP. Among them, XGBoost
    performed best on the held-out test set, achieving the highest Accuracy, F1 score, and AUC-ROC. This indicates that the relationship
    between engagement characteristics and AI usage is likely nonlinear and better captured by boosted tree ensembles than by simpler baselines.
    """)

    st.write("""
    The broader conclusion is that AI usage in audit settings can be modeled as a structured business prediction task, but model choice matters.
    Simpler models offer interpretability, while the strongest predictive performance in this notebook came from XGBoost. The explainability
    tab complements performance metrics by showing how feature patterns influence individual predictions.
    """)

# =============================
# TAB 2: Descriptive Analytics
# =============================
with tab2:
    st.title("Descriptive Analytics")

    show_image_with_analysis(
        "outputs/target_distribution.png",
        "Target Distribution: AI Used for Auditing",
        """
        The target distribution shows that the classes are reasonably balanced, with slightly more non-AI than AI observations.
        This is useful because the modeling problem does not appear to be dominated by severe class imbalance, so the reported Accuracy,
        F1, and AUC-ROC can be interpreted in a fairly standard way. A balanced target also reduces the likelihood that a model can look
        strong simply by predicting the majority class.
        """
    )

    show_image_with_analysis(
        "outputs/viz_1.png",
        "Audit Effectiveness Score by AI Usage",
        """
        This visualization compares audit effectiveness across AI-enabled and non-AI cases. The main purpose is to evaluate whether AI use
        appears to be associated with stronger audit-quality outcomes. If the AI group shows a higher or tighter distribution, that suggests
        AI may be linked to more consistent or more effective audit execution rather than being applied randomly across engagements.
        """
    )

    show_image_with_analysis(
        "outputs/viz_2.png",
        "Employee Workload by AI Usage",
        """
        This plot evaluates whether AI adoption is associated with higher or lower employee workload. If AI usage is more common in
        higher-workload cases, that suggests firms may be using AI in response to operational pressure and complexity. If workload is lower
        for AI cases, the interpretation would instead lean toward efficiency benefits. Either way, workload appears to be a practically
        meaningful business feature rather than a background variable.
        """
    )

    show_image_with_analysis(
        "outputs/viz_3.png",
        "AI Usage Rate by Industry",
        """
        This chart examines whether AI adoption differs across industries. Industry-level variation matters because some sectors are more
        regulation-heavy, data-intensive, or operationally complex than others. If the AI usage rate is visibly higher in certain industries,
        that supports the idea that industry context is an important driver of audit technology adoption and justifies keeping it in the model.
        """
    )

    show_image_with_analysis(
        "outputs/viz_4.png",
        "High-Risk Cases vs Fraud Cases Detected",
        """
        This scatter plot helps reveal whether AI-enabled auditing is more common in complex engagements with both elevated risk and more
        fraud-detection activity. If AI cases cluster in the upper-right region of the chart, that suggests technology adoption is more likely
        in engagements that are harder to monitor manually. This is a meaningful business pattern because it links AI usage to actual audit
        complexity rather than to arbitrary firm-level preference alone.
        """
    )

    show_image_with_analysis(
        "outputs/correlation_heatmap.png",
        "Correlation Heatmap",
        """
        The heatmap summarizes the strongest linear relationships among the numerical variables. It is useful for identifying whether
        variables such as workload, risk exposure, fraud cases, revenue impact, effectiveness, and satisfaction move together. From a
        modeling perspective, this helps flag overlapping information and potential multicollinearity, which may weaken linear models but
        is typically less problematic for tree-based methods.
        """
    )

# =============================
# TAB 3: Model Performance
# =============================
with tab3:
    st.title("Model Performance")

    st.subheader("Model Comparison Table")
    st.dataframe(results_df, use_container_width=True)

    st.write("""
    The model comparison table summarizes the full test-set evaluation across all five models. XGBoost delivered the strongest overall
    performance, with Accuracy = 0.600, F1 = 0.538, and AUC-ROC = 0.647. Logistic Regression and MLP performed at a middle level, while
    Decision Tree and Random Forest underperformed on this dataset. This pattern suggests that nonlinear structure exists in the data, but
    it is captured more effectively by gradient boosting than by a single tree or a basic bagging ensemble.
    """)

    show_image_with_analysis(
        "outputs/model_comparison.png",
        "Model Comparison by F1 Score",
        """
        The bar chart makes the ranking of the candidate models visually clear. XGBoost has the highest F1 score, followed by MLP and
        Logistic Regression, while Decision Tree and Random Forest lag behind. Because F1 balances precision and recall, this chart indicates
        that XGBoost achieved the best compromise between identifying AI-enabled cases and avoiding too many false positives.
        """
    )

    show_image_with_analysis(
        "outputs/roc_logistic.png",
        "ROC Curve - Logistic Regression",
        """
        Logistic Regression served as the baseline model. Its AUC-ROC was approximately 0.498, which is close to random guessing.
        This indicates that a simple linear decision boundary is not sufficient to capture the structure of AI adoption in this dataset.
        """
    )

    show_image_with_analysis(
        "outputs/roc_tree.png",
        "ROC Curve - Decision Tree",
        """
        The Decision Tree achieved Accuracy = 0.400, F1 = 0.308, and AUC-ROC = 0.457, with best parameters max_depth = 3 and
        min_samples_leaf = 10. These results show that the shallow tree was not able to generalize strongly on the held-out test set,
        even after cross-validated tuning. It remains interpretable, but its predictive performance was weak.
        """
    )

    show_image_with_analysis(
        "outputs/roc_rf.png",
        "ROC Curve - Random Forest",
        """
        Random Forest achieved Accuracy = 0.367, F1 = 0.240, and AUC-ROC = 0.466, with best parameters max_depth = 8 and
        n_estimators = 100. Despite being a more flexible ensemble than a single tree, it did not perform well here. That may reflect the
        small dataset size, limited signal, or the fact that the strongest patterns are better captured by boosting than by bagging.
        """
    )

    show_image_with_analysis(
        "outputs/roc_xgb.png",
        "ROC Curve - XGBoost",
        """
        XGBoost was the best-performing model in the notebook, with Accuracy = 0.600, Precision = 0.538, Recall = 0.538,
        F1 = 0.538, and AUC-ROC = 0.647. Its best hyperparameters were learning_rate = 0.1, max_depth = 4, and
        n_estimators = 200. This result suggests that boosted trees were best able to capture the nonlinear interactions in the audit dataset.
        """
    )

    show_image_with_analysis(
        "outputs/mlp_history.png",
        "MLP Training Loss Curve",
        """
        The MLP training curve shows how the neural network optimized its loss over training epochs. On the test set, the MLP achieved
        Accuracy = 0.533, F1 = 0.462, and AUC-ROC = 0.462. This means it performed better than the tree-based baselines except XGBoost,
        but it still did not surpass the boosted tree model. On a small tabular dataset like this one, that outcome is not surprising.
        """
    )

    st.subheader("Best Hyperparameters")
    st.write("**Decision Tree:** max_depth = 3, min_samples_leaf = 10")
    st.write("**Random Forest:** max_depth = 8, n_estimators = 100")
    st.write("**XGBoost:** learning_rate = 0.1, max_depth = 4, n_estimators = 200")

    st.write("""
    In terms of trade-offs, Logistic Regression remains the simplest and most interpretable baseline but was too weak for this dataset.
    Decision Tree is easy to explain visually but performed poorly. Random Forest was more robust in theory yet still weak here.
    MLP introduced more flexibility but did not outperform XGBoost. XGBoost provided the strongest predictive performance, but it is also
    less transparent without an explainability layer, which is why SHAP is especially important for the final interpretation.
    """)

# =============================
# TAB 4: Explainability & Interactive Prediction
# =============================
with tab4:
    st.title("Explainability & Interactive Prediction")

    st.subheader("SHAP Explainability")

    show_image_with_analysis(
        "outputs/shap_summary.png",
        "SHAP Summary Plot",
        """
        The SHAP summary plot shows both feature importance and direction of effect across the test set. Features appearing near the top
        have the largest overall contribution to the XGBoost predictions. The plot helps identify whether high or low feature values push
        the prediction toward AI usage. This is important because it turns the best-performing model from a black box into an interpretable
        decision-support tool.
        """
    )

    show_image_with_analysis(
        "outputs/shap_bar.png",
        "SHAP Bar Plot",
        """
        The SHAP bar plot ranks features by average absolute contribution. In this project, the most influential variables are the ones
        at the top of that ranking, meaning they drive the model’s decisions most strongly on average. For a business audience, this plot
        is especially useful because it answers the question: which engagement characteristics matter most when predicting AI adoption?
        """
    )

    show_image_with_analysis(
        "outputs/shap_waterfall.png",
        "SHAP Waterfall Plot",
        """
        The waterfall plot explains one individual prediction by showing how each feature pushes the model output upward or downward from
        the baseline prediction. This is useful for case-level interpretation because it connects a single engagement’s profile to the
        final prediction in a transparent way. In practice, a decision-maker could use this to understand why a case is predicted to be
        AI-enabled and which specific factors contributed most to that result.
        """
    )

    st.write("""
    Because XGBoost was the strongest model in the notebook, it was the appropriate choice for SHAP analysis. The SHAP outputs are useful
    not only for explanation but also for managerial action: they help identify whether AI adoption is being driven more by operational load,
    fraud and risk patterns, industry context, or audit-quality indicators.
    """)

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
    total_eng = st.slider(
        "Total Audit Engagements",
        int(df["Total_Audit_Engagements"].min()),
        int(df["Total_Audit_Engagements"].max()),
        int(df["Total_Audit_Engagements"].median())
    )
    high_risk = st.slider(
        "High Risk Cases",
        int(df["High_Risk_Cases"].min()),
        int(df["High_Risk_Cases"].max()),
        int(df["High_Risk_Cases"].median())
    )
    violations = st.slider(
        "Compliance Violations",
        int(df["Compliance_Violations"].min()),
        int(df["Compliance_Violations"].max()),
        int(df["Compliance_Violations"].median())
    )
    fraud = st.slider(
        "Fraud Cases Detected",
        int(df["Fraud_Cases_Detected"].min()),
        int(df["Fraud_Cases_Detected"].max()),
        int(df["Fraud_Cases_Detected"].median())
    )
    revenue = st.slider(
        "Total Revenue Impact",
        float(df["Total_Revenue_Impact"].min()),
        float(df["Total_Revenue_Impact"].max()),
        float(df["Total_Revenue_Impact"].median())
    )
    workload = st.slider(
        "Employee Workload",
        int(df["Employee_Workload"].min()),
        int(df["Employee_Workload"].max()),
        int(df["Employee_Workload"].median())
    )
    effectiveness = st.slider(
        "Audit Effectiveness Score",
        float(df["Audit_Effectiveness_Score"].min()),
        float(df["Audit_Effectiveness_Score"].max()),
        float(df["Audit_Effectiveness_Score"].median())
    )
    satisfaction = st.slider(
        "Client Satisfaction Score",
        float(df["Client_Satisfaction_Score"].min()),
        float(df["Client_Satisfaction_Score"].max()),
        float(df["Client_Satisfaction_Score"].median())
    )
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

    if selected_model is not None:
        pred_prob = selected_model.predict_proba(input_df)[0, 1]
        pred_class = "Yes" if pred_prob >= 0.5 else "No"

        st.write(f"### Predicted AI Usage: {pred_class}")
        st.write(f"### Predicted Probability of AI Usage: {pred_prob:.3f}")

        st.write("""
        This interactive section allows the user to change engagement characteristics and immediately observe how the predicted probability
        of AI-enabled auditing changes. It turns the project from a static report into a practical decision-support tool.
        """)
    else:
        st.warning("The selected model file is not available yet.")
