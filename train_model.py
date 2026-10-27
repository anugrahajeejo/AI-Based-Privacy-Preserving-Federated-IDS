
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Step 1: Create simulated network traffic data
# 0 = Normal traffic
# 1 = Suspicious traffic

X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=10,
    n_redundant=4,
    n_classes=2,
    weights=[0.7, 0.3],
    random_state=42
)

print("Dataset created successfully!")
print("Total records:", len(X))
print("Number of features:", X.shape[1])

# Step 2: Divide data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Step 3: Create the ML model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# Step 4: Train the model

model.fit(X_train, y_train)

print("\nModel training completed!")

# Step 5: Predict test records

y_pred = model.predict(X_test)

# Step 6: Evaluate the model

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Suspicious"]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nPhase 1 completed successfully!")
