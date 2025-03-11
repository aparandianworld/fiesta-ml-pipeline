import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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

# Train, predict, and evaluate
try: 
    pipeline.fit(X_train, y_train)
    y_hat = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_hat)
    print(f"Test accuracy: {accuracy:.2f}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
