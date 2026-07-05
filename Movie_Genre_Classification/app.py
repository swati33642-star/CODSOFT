import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Title
st.title("🎬 Movie Genre Classification")

st.write("Enter a movie description to predict its genre.")

# Input
description = st.text_area("Movie Description")

# Prediction
if st.button("Predict Genre"):

    if description.strip() == "":
        st.warning("Please enter a movie description.")
    else:
        text = vectorizer.transform([description])
        prediction = model.predict(text)

        st.success(f"🎥 Predicted Genre: {prediction[0]}")