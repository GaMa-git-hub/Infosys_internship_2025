import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris

# ---------------------------
# Load trained model and dataset
# ---------------------------
model = joblib.load("iris_model.pkl")  # Make sure iris_model.pkl is in the same folder
iris = load_iris(as_frame=True)
df = iris.frame

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("🔎 Navigation")
mode = st.sidebar.radio("Choose mode:", ["Prediction", "Data Exploration"])

# ---------------------------
# Prediction Mode
# ---------------------------
if mode == "Prediction":
    st.title("🌸 Iris Flower Prediction App")
    st.write("Enter the flower features below and get a prediction using the trained model.")

    # Input widgets with tooltips
    sepal_length = st.slider(
        "Sepal Length (cm)",
        float(df["sepal length (cm)"].min()),
        float(df["sepal length (cm)"].max()),
        float(df["sepal length (cm)"].mean()),
        help="Length of the sepal in centimeters."
    )

    sepal_width = st.slider(
        "Sepal Width (cm)",
        float(df["sepal width (cm)"].min()),
        float(df["sepal width (cm)"].max()),
        float(df["sepal width (cm)"].mean()),
        help="Width of the sepal in centimeters."
    )

    petal_length = st.slider(
        "Petal Length (cm)",
        float(df["petal length (cm)"].min()),
        float(df["petal length (cm)"].max()),
        float(df["petal length (cm)"].mean()),
        help="Length of the petal in centimeters."
    )

    petal_width = st.slider(
        "Petal Width (cm)",
        float(df["petal width (cm)"].min()),
        float(df["petal width (cm)"].max()),
        float(df["petal width (cm)"].mean()),
        help="Width of the petal in centimeters."
    )

    # Prepare input as DataFrame using only feature columns
    feature_columns = iris.feature_names
    features = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=feature_columns
    )

    # Predict
    prediction = model.predict(features)[0]

    # Display result
    st.subheader("🌼 Prediction Result")
    st.success(f"The predicted species is: **{iris.target_names[prediction]}**")

# ---------------------------
# Data Exploration Mode
# ---------------------------
else:
    st.title("📊 Iris Dataset Explorer")
    st.write("Explore the Iris dataset with simple charts.")

    # Histogram using Streamlit native chart
    st.subheader("🔹 Feature Distribution (Sepal Length)")
    st.bar_chart(df["sepal length (cm)"])

    # Scatter plot using Streamlit native chart
    st.subheader("🔹 Sepal Length vs Sepal Width")
    st.scatter_chart(df[["sepal length (cm)", "sepal width (cm)"]])

    st.subheader("🔹 Petal Length vs Petal Width")
    st.scatter_chart(df[["petal length (cm)", "petal width (cm)"]])
