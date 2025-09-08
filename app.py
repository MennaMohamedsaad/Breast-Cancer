import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# App title
st.title("🎗️ Breast Cancer Prediction App")

# Load dataset
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

# Show data
st.subheader("🔹 Dataset Preview")
st.write(df.head())

# Show basic statistics
st.subheader("🔹 Dataset Description")
st.write(df.describe())

# Split dataset
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
st.subheader("🔹 Model Accuracy")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Confusion Matrix
st.subheader("🔹 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cancer.target_names)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
st.pyplot(fig)

# Classification report
st.subheader("🔹 Classification Report")
st.text(classification_report(y_test, y_pred, target_names=cancer.target_names))
