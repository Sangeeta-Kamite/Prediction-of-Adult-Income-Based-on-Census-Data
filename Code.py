# Import necessary libraries
pip install pandas numpy scikit-learn matplotlib seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]
df = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)

# Display dataset overview
print(df.head())
print("\nDataset Info:\n", df.info())

# Handling missing values
df.dropna(inplace=True)

# Encoding categorical features
categorical_columns = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country"
]

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target variable
df["income"] = df["income"].apply(lambda x: 1 if x == ">50K" else 0)

# Splitting dataset into features and target
X = df.drop("income", axis=1)
y = df["income"]

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Display confusion matrix for best-performing model
best_model = models["Random Forest"]
y_pred = best_model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["≤50K", ">50K"], yticklabels=["≤50K", ">50K"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Predicting income for a new person
def predict_income(new_data):
    new_df = pd.DataFrame([new_data], columns=X.columns)
    for col, le in label_encoders.items():
        new_df[col] = le.transform([new_df[col].values[0]])
    new_df = scaler.transform(new_df)
    prediction = best_model.predict(new_df)
    return "Income > $50K" if prediction[0] == 1 else "Income ≤ $50K"

# Example prediction
new_person = {
    "age": 35, "workclass": "Private", "fnlwgt": 150000, "education": "Bachelors",
    "education_num": 13, "marital_status": "Married-civ-spouse", "occupation": "Exec-managerial",
    "relationship": "Husband", "race": "White", "sex": "Male",
    "capital_gain": 5000, "capital_loss": 0, "hours_per_week": 40, "native_country": "United-States"
}

print("\nPrediction for new individual:", predict_income(new_person))

