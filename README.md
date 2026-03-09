# Predicting AI Adoption in Big Four Auditing
Can we predict whether a Big Four audit/risk/compliance case will use AI-enabled auditing methods based on engagement volume, risk exposure, fraud detection, revenue impact, industry, workload, and quality/satisfaction indicators?

## Overview
This project builds an end-to-end machine learning workflow on a tabular auditing and compliance dataset.
The target variable is `AI_Used_for_Auditing`.

## Repository Contents
- `notebooks/` : Colab / notebook analysis
- `models/` : saved trained models
- `outputs/` : charts and SHAP plots
- `app.py` : Streamlit app
- `requirements.txt` : dependencies

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Run the app:
   streamlit run app.py

## Deployment
The app is deployed publicly via Streamlit Community Cloud.
