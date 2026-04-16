import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# 1. Data Preparation
# Note: If you have a real CSV file, use: df = pd.read_csv('your_file.csv')
# Here we generate synthetic data representing "Telecom Churn" (Imbalanced 80/20)
X_raw, y = make_classification(
    n_samples=1500, n_features=12, n_informative=8, 
    n_clusters_per_class=1, weights=[0.8, 0.2], random_state=42
)
X = pd.DataFrame(X_raw)

# 2. Feature Scaling
# Essential for Logistic Regression convergence and performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Model & Cross-Validation Setup
# Using StratifiedKFold to maintain class balance in each fold (Constraint Requirement)
cv = StratifiedKFold(n_splits=5)
model = LogisticRegression(max_iter=1000)

# 4. Generate Learning Curve
# Using 'f1' scoring instead of 'accuracy' due to data imbalance (Constraint Requirement)
train_sizes, train_scores, test_scores = learning_curve(
    model, X_scaled, y, 
    cv=cv, 
    scoring='f1', 
    train_sizes=np.linspace(0.1, 1.0, 5), # 5 different training set sizes
    n_jobs=-1
)

# 5. Calculate Mean and Standard Deviation
# Required for plotting the shaded uncertainty area
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# 6. Visualization
plt.figure(figsize=(10, 6))

# Plot mean scores
plt.plot(train_sizes, train_mean, 'o-', color="red", label="Training Score (F1)")
plt.plot(train_sizes, test_mean, 'o-', color="green", label="Cross-validation Score (F1)")

# Plot Standard Deviation Shading (Constraint Requirement)
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="red")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="green")

# Chart metadata
plt.title("Learning Curve Diagnosis (Telecom Churn Analysis)")
plt.xlabel("Training Set Size")
plt.ylabel("F1 Score")
plt.legend(loc="best")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('learning_curve_plot.png')
print("Plot saved as learning_curve_plot.png")