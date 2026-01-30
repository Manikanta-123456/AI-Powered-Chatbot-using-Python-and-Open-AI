import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load dataset (assuming preprocessed CSV files inside UCI HAR Dataset folder)
X_train = pd.read_csv('UCI HAR Dataset/X_train.txt', delim_whitespace=True, header=None)
y_train = pd.read_csv('UCI HAR Dataset/y_train.txt', delim_whitespace=True, header=None)
X_test = pd.read_csv('UCI HAR Dataset/X_test.txt', delim_whitespace=True, header=None)
y_test = pd.read_csv('UCI HAR Dataset/y_test.txt', delim_whitespace=True, header=None)

# Initialize Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train.values.ravel())

# Predict
y_pred = clf.predict(X_test)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))
