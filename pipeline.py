import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Pipeline
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2)),
        ("classifier", LogisticRegression()),
    ]
)

# Parameter grid
param_grid = {
    "pca__n_components": [2, 3, 4],
    "classifier__C": [0.1, 1, 10],
    "classifier__solver": ["liblinear", "sag", "liblinear"],
    "classifier__max_iter": [1000],
}

# Grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy")

# Train, predict, and evaluate
try:
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross validation score: {grid_search.best_score_:.2f}")
    y_hat = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_hat)
    print(f"Test accuracy with best model: {test_accuracy:.2f}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
