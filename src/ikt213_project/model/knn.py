from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from pathlib import Path


project_root = Path(__file__).resolve().parents[3]
data_path = project_root / 'data'

# Load extracted landmarks
cols = ["label"] + [f"{axis}{i}" for i in range(21) for axis in "xyz"]
df = pd.read_csv(data_path / "extractedLandmarks.csv", header=None, names=cols) 

X = df.drop(columns="label")
y = df["label"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Train the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
print(f"Test Accuracy (k=5): {accuracy_score(y_test, y_pred):.2f}")

print(df["label"].unique())
print(df["label"].nunique())

