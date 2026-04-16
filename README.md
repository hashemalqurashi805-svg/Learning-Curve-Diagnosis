# Learning Curve Diagnosis Project

This project implements a **Learning Curve Diagnosis** for a Telecom Churn classification task using **Logistic Regression**. The goal is to analyze the model's performance to identify Bias-Variance tradeoffs.

## Project Overview
The project uses `scikit-learn` to generate learning curves with the following constraints:
- **Stratified K-Fold Cross-Validation** (5 folds).
- **F1-Score** as the primary evaluation metric (due to class imbalance).
- **Standard Deviation Shading** to visualize model stability.

## Files in this Repository
- `main.py`: The core Python script that handles data scaling, model training, and plotting.
- `analysis.md`: A detailed technical diagnosis of the model's behavior based on the results.
- `learning_curve_plot.png`: The visual output of the learning curve diagnosis.
- `requirements.txt`: List of necessary Python libraries.

## How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/hashemalqurashi805-svg/Learning-Curve-Diagnosis.git](https://github.com/hashemalqurashi805-svg/Learning-Curve-Diagnosis.git)