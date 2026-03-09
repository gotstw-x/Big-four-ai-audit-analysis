import os
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="HW1 - Big4 AI Auditing", layout="wide")

# =========================================================
# Helper functions
# =========================================================

def file_exists(path: str) -> bool:
    return os.path.exists(path)

def safe_read_csv(path: str, fallback_df: pd.DataFrame | None = None) -> pd.DataFrame | None:
    try:
        if file_exists(path):
            return pd.read_csv(path)
        return fallback_df
    except Exception as e:
        st.warning(f"Could not read CSV file: {path}")
        st.exception(e)
        return fallback_df

def safe_load_joblib(path: str):
    try:
        if file_exists(path):
            return joblib.load(path)
        return None
    except Exception as e:
        st.warning(f"Could not load model file: {path}")
        st.exception(e)
        return None

def show_image_if_exists(path: str, caption: str):
    if file_exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing image file: {path}")

def get_prediction_probability(model, input_df: pd.DataFrame):
    """
    Returns predicted class label and probability safely.
    Works best for classifiers with predict_proba().
    """
    try:
        if model is None:
            return None, None, "The selected model is not available."

        # Preferred path: predict_proba
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(input_df)[0, 1])
            pred = "Yes" if prob >= 0.5 else "No"
            return pred, prob, None

        # Fallback: decision_function
        if hasattr(model, "decision_function"):
            score = float(model.decision_function(input_df)[0])
            # crude sigmoid fallback for display only
            import math
            prob = 1 / (1 + math.exp(-score))
            pred = "Yes" if prob >= 0.5 else "No"
            return pred, prob, None

        # Final fallback: predict only
        pred_raw = model.predict(input_df)[0]
        if str(pred_raw) in ["1", "Yes", "yes", "True", "true"]:
            return "Yes", None, None
        return "No", None, None

    except Exception as e:
        return None, None, f"Prediction failed: {e}"

def safe_metric_table():
    return pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "MLP"],
        "Accuracy": [None] * 5,
        "Precision": [None] * 5,
        "Recall": [None] * 5,
        "F1": [None] * 5,
        "AUC_ROC": [None] * 5
    })

# =========================================================
# Paths
# =========================================================

DATA_PATH = "data/big4_financial_risk_compliance.csv"
RESULTS_PATH = "models/model_results.csv"

LOG_MODEL_PATH = "models/logistic_model.joblib"
TREE_MODEL_PATH = "models/decision_tree_model.joblib"
RF_MODEL_PATH = "models/random_forest_model.joblib"
XGB_MODEL_PATH = "models/xgb_model.joblib"

IMG_TARGET = "outputs/target_distribution.png"
IMG_VIZ1 = "outputs/viz_1.png"
IMG_VIZ2 = "outputs/viz_2.png"
IMG_VIZ3 = "outputs/viz_3.png"
IMG_VIZ4 = "outputs/viz_4.png"
IMG_HEATMAP = "outputs/correlation_heatmap.png"

IMG_MODEL_COMPARE = "outputs/model_comparison.png"
IMG_ROC_LOG = "outputs/roc_logistic.png"
IMG_ROC_TREE = "outputs/roc_tree.png"
IMG_ROC_RF = "outputs/roc_rf.png"
IMG_ROC_XGB = "outputs/roc_xgb.png"
IMG_MLP = "outputs/mlp_history.png"

IMG_SHAP_SUMMARY = "outputs/shap_summary.png"
IMG_SHAP_BAR = "outputs/shap_bar.png"
IMG_SHAP_WATERFALL = "outputs/shap_waterfall.png"

# =========================================================
# Load data and assets safely
# =========================================================

df = safe_read_csv(DATA_PATH)

if df is None:
    st.error("The main dataset file is missing: data/big4_financial_risk_compliance.csv")
    st.stop()

results_df = safe_read_csv(RESULTS_PATH, fallback_df=safe_metric_table())
if results_df is None:
    results_df = safe_metric_table()

log_model = safe_load_joblib(LOG_MODEL_PATH)
tree_model = safe_load_joblib(TREE_MODEL_PATH)
rf_model = safe_load_joblib(RF_MODEL_PATH)
xgb_model = safe_load_joblib(XGB_MODEL_PATH)

target = "AI_Used_for_Auditing"

# =========================================================
# Sidebar diagnostics
# =========================================================

with st.sidebar:
    st.header("App Diagnostics")

    st.write("**Dataset**")
    st.write("Loaded" if df is not None else "Missing")

    st.write("**Models**")
    st.write(f"Logistic Regression: {'Loaded' if log_model is not None else 'Missing'}")
    st.write(f"Decision Tree: {'Loaded' if tree_model is not None else 'Missing'}")
    st.write(f"Random Forest: {'Loaded' if rf_model is not None else 'Missing'}")
    st.write(f"XGBoost: {'Loaded' if xgb_model is not None else 'Missing'}")

    st.write("**Results Table**")
    st.write("Loaded" if file_exists(RESULTS_PATH) else "Fallback table in use")

# =========================================================
# Tabs
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Interactive Prediction"
])

# =========================================================
# Tab 1
# =========================================================
with tab1:
    st.title("Executive Summary")

    st.write("""
    This project predicts whether AI was used for auditing in a Big Four financial risk and compliance dataset.
    The dataset contains engagement-level operational, risk, fraud, industry, workload, and audit quality variables.
    The target variable is `AI_Used_for_Auditing`, which indicates whether AI-enabled auditing was used in the case.
    """)

    st.write("""
    This problem is meaningful because AI adoption in auditing affects efficiency, consistency, staffing strategy,
    fraud detection capability, and digital transformation planning. Understanding what conditions are associated
    with AI-enabled auditing can help firms allocate resources and technology more effectively.
    """)

    st.write("""
    The workflow includes descriptive analytics, preprocessing, multiple classification models, SHAP-based explainability,
    and an interactive Streamlit deployment. The app is designed to summarize the entire HW1 pipeline in one place.
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

# =========================================================
# Tab 2
# =========================================================
with tab2:
    st.title("Descriptive Analytics")

    show_image_if_exists(
        IMG_TARGET,
        "The target distribution is relatively balanced, which supports stable classification modeling."
    )

    show_image_if_exists(
        IMG_VIZ1,
        "This figure compares audit effectiveness across AI and non-AI cases."
    )

    show_image_if_exists(
        IMG_VIZ2,
        "This figure shows how employee workload differs between AI and non-AI auditing cases."
    )

    show_image_if_exists(
        IMG_VIZ3,
        "This chart compares AI adoption rates across industries."
    )

    show_image_if_exists(
        IMG_VIZ4,
        "This plot shows the relationship between high-risk cases and fraud cases detected, colored by AI usage."
    )

    show_image_if_exists(
        IMG_HEATMAP,
        "The heatmap summarizes linear relationships among numerical variables."
    )

# =========================================================
# Tab 3
# =========================================================
with tab3:
    st.title("Model Performance")

    st.subheader("Model Comparison Table")
    st.dataframe(results_df, use_container_width=True)

    st.write("""
    This table summarizes the predictive performance of all trained models.
    Accuracy gives the overall proportion of correct predictions, while precision and recall evaluate the quality of positive predictions from different angles.
    F1 score is particularly useful here because it balances precision and recall, and AUC-ROC measures how well the model separates the two classes across different thresholds.
    """)

    show_image_if_exists(
        IMG_MODEL_COMPARE,
        "This bar chart compares all models using the selected primary metric."
    )
    st.caption("""
    This chart makes it easier to compare model performance visually.
    A higher score indicates better predictive performance under the chosen evaluation metric.
    If one of the tree-based ensemble models performs best, that suggests nonlinear feature interactions are important in explaining AI adoption.
    """)

    show_image_if_exists(IMG_ROC_LOG, "ROC curve for Logistic Regression.")
    st.caption("""
    Logistic Regression serves as the baseline model.
    If its ROC curve is clearly below the tree-based models, that suggests the relationship between predictors and AI usage is not purely linear.
    """)

    show_image_if_exists(IMG_ROC_TREE, "ROC curve for Decision Tree.")
    st.caption("""
    The Decision Tree model captures nonlinear splits and interaction effects.
    Its performance helps show whether threshold-style decision rules are useful for this dataset.
    """)

    show_image_if_exists(IMG_ROC_RF, "ROC curve for Random Forest.")
    st.caption("""
    Random Forest improves stability by averaging predictions across many trees.
    If this model outperforms the single Decision Tree, it indicates that ensembling reduces overfitting and improves generalization.
    """)

    show_image_if_exists(IMG_ROC_XGB, "ROC curve for XGBoost.")
    st.caption("""
    XGBoost is often one of the strongest methods for structured tabular data.
    If this curve is the highest among all models, it suggests that boosting is especially effective for capturing subtle predictive patterns in the audit dataset.
    """)

    show_image_if_exists(IMG_MLP, "Training loss curve for the PyTorch MLP.")
    st.caption("""
    The loss curve shows how the neural network learned during training.
    A downward trend suggests that the model was able to reduce prediction error over epochs, while instability or flatness may indicate limited signal or insufficient data volume.
    """)

    st.subheader("Best Hyperparameters")
    st.info("Replace this section with the final best hyperparameters from Colab after tuning.")

    st.subheader("Model Interpretation Summary")
    st.write("""
    The model comparison suggests which method best captures the relationship between operational audit variables and AI usage.
    If the best-performing models are Random Forest or XGBoost, the evidence supports the presence of nonlinearities and feature interactions.
    If Logistic Regression performs similarly to the more complex models, then the decision boundary may be relatively simple and easier to interpret.
    """)
# =========================================================
# Tab 4
# =========================================================
with tab4:
    st.title("Explainability & Interactive Prediction")

    st.subheader("SHAP Explainability")
    st.write("""
    SHAP values help explain how the model arrives at its predictions.
    They quantify how much each feature pushes the prediction toward either AI usage or non-AI usage.
    This improves transparency by showing which operational and audit-related variables drive the model output.
    """)

    show_image_if_exists(IMG_SHAP_SUMMARY, "SHAP summary plot.")
    st.caption("""
    The SHAP summary plot ranks features by overall importance and also shows the direction of each feature's influence.
    Features appearing near the top have the strongest impact on the model across all observations.
    This plot is useful for identifying the main drivers of predicted AI adoption.
    """)

    show_image_if_exists(IMG_SHAP_BAR, "SHAP bar plot.")
    st.caption("""
    The SHAP bar plot summarizes the average absolute importance of each feature.
    It is a simpler way to compare feature influence without showing individual point-level variation.
    This chart helps stakeholders quickly identify the most influential predictors.
    """)

    show_image_if_exists(IMG_SHAP_WATERFALL, "SHAP waterfall plot for one sample prediction.")
    st.caption("""
    The waterfall plot explains one specific prediction step by step.
    It starts from the model's baseline prediction and shows how each feature pushes the final result upward or downward.
    This is especially useful for understanding individual cases rather than the dataset as a whole.
    """)

    st.subheader("Interactive Prediction")
    st.write("""
    The interactive prediction tool allows the user to manually adjust input values and observe how the selected model responds.
    This feature makes the analysis more practical by turning the trained model into a decision-support interface.
    It also helps demonstrate how operational factors may change the predicted probability of AI usage.
    """)
