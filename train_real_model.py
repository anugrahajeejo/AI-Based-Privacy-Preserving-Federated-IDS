import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Loading CICIDS2017 dataset...")

# Load the dataset
df = pd.read_csv("data/combine.csv", low_memory=False)

print("Dataset loaded:", df.shape)

# Clean column names
df.columns = df.columns.str.strip()

# Remove spaces from labels
df["Label"] = df["Label"].astype(str).str.strip()

# Convert labels into two classes
# 0 = Normal traffic
# 1 = Attack traffic
y = (df["Label"] != "BENIGN").astype(int)

# Select numerical network traffic features
X = df.drop(columns=["Label"])

# Convert all features to numeric values
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinite values with missing values
X = X.replace([np.inf, -np.inf], np.nan)

# Replace missing values with zero
X = X.fillna(0)

# Remove features that contain no useful variation
X = X.loc[:, X.nunique() > 1]

print("Features prepared:", X.shape[1])

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Create the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("Training Random Forest model...")
model.fit(X_train, y_train)

print("Training completed!")

# Evaluate the model
y_pred = model.predict(X_test)

print("\nAccuracy:", round(
    accuracy_score(y_test, y_pred) * 100, 2
), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Attack"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nReal-data baseline completed!")
# Save the trained model and its feature names
joblib.dump(model, "ids_model.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print("Model saved successfully!")
