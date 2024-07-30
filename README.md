# Fraud-Detection-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Display the first few rows of the dataset
print(df.head())

# Display the column names
print(df.columns)

# Check for any missing values
print(df.isnull().sum())

# Statistical summary of the dataset
print(df.describe())

# Define the target and features
target_column = 'class'  # Fraudulent transactions are labeled as '1' in the 'Class' column

# Split the data into features (X) and target (y)
X = df.drop(target_column, axis=1)
y = df[target_column]

# Use a subset of the data to reduce time for testing purposes
# Uncomment the following line to use a subset
# X, y = X.sample(10000, random_state=42), y.sample(10000, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature columns
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train a Random Forest Classifier with fewer estimators for quicker runs
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train_res, y_train_res)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

This code is a Python script that uses various libraries (pandas, NumPy, scikit-learn, and imbalanced-learn) to train a Random Forest Classifier model on a credit card fraud detection dataset. Here's a breakdown of the code:

Importing Libraries

The script starts by importing the necessary libraries:

- `pandas` (pd) for data manipulation and analysis
- `NumPy` (np) for numerical computations
- `sklearn` for machine learning tasks (model selection, preprocessing, metrics)
- `imblearn` for handling class imbalance (SMOTE oversampling)

Loading and Exploring the Dataset

- Loads the credit card fraud detection dataset from a CSV file named `creditcard.csv` into a pandas DataFrame (`df`).
- Displays the first few rows of the dataset using `df.head()`.
- Prints the column names using `df.columns`.
- Checks for missing values using `df.isnull().sum()`.
- Prints a statistical summary of the dataset using `df.describe()`.

Preparing the Data

- Defines the target column (`class`) and splits the data into features (`X`) and target (`y`).
- Optionally, reduces the dataset size for testing purposes by uncommenting the sampling lines.
- Splits the data into training and testing sets using `train_test_split` (80% for training and 20% for testing).

Data Preprocessing

- Standardizes the feature columns using `StandardScaler` to ensure all features are on the same scale.
- Handles class imbalance using SMOTE oversampling on the training data.

Training the Model

- Trains a Random Forest Classifier model with 50 estimators (trees) on the resampled training data.
- Sets `n_jobs=-1` to use all available CPU cores for parallel processing.

Evaluating the Model

- Makes predictions on the test set using the trained model.
- Prints evaluation metrics:
    - Confusion Matrix
    - Classification Report (precision, recall, F1-score, support)
    - Accuracy Score

The goal of this script is to train a machine learning model to detect fraudulent credit card transactions based on the features in the dataset. The model is evaluated using various metrics to assess its performance.
