# -B-13-PrognosAI-AI-Driven-Predictive-Maintenance-System
🚀 Engine Remaining Useful Life (RUL) Prediction System

📌 Project Overview

This project predicts the Remaining Useful Life (RUL) of engines using Machine Learning and Deep Learning techniques.
The goal is to detect possible engine failure early so that maintenance can be done in advance, reducing cost and avoiding unexpected breakdowns.

🎯 Problem Statement

In industries like aviation and manufacturing, engine failure can lead to:
High maintenance cost
Unexpected downtime
Safety risks
This project solves the problem by predicting how long an engine can still operate before failure.

💡 Solution Approach

The system follows a complete pipeline:
Data preprocessing and cleaning
Feature engineering and RUL calculation
Baseline model comparison
Deep learning model (LSTM) for prediction
Model evaluation using metrics
Alert classification (Healthy, Warning, Critical)
Interactive dashboard for visualization

⚙️ Technologies Used

Python
Pandas, NumPy → Data processing
Scikit-learn → Baseline models
TensorFlow / Keras → LSTM model
Matplotlib → Visualization
Streamlit → Dashboard

📊 Models Implemented

🔹 Baseline Models
Linear Regression
Random Forest
XGBoost
These models were used for comparison.
🔹 Final Model
LSTM (Long Short-Term Memory)
Used to capture time-series behavior of engine data.

📈 Model Evaluation Metrics

The model performance was evaluated using:
RMSE (Root Mean Squared Error) → measures prediction error
MAE (Mean Absolute Error) → average error
R² Score → model accuracy

🚨 Alert Classification

Based on predicted RUL, engines are classified into:
🟢 Healthy → RUL ≥ 60
🟠 Warning → 30 ≤ RUL < 60
🔴 Critical → RUL < 30
This helps in prioritizing maintenance decisiON


🌍 Real-World Applications

This system can be used in:
✈️ Aircraft engine monitoring
🏭 Industrial machine maintenance
🚗 Automotive systems
⚙️ Predictive maintenance systems


