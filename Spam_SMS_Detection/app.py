import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Spam SMS Detector",
    page_icon="📩",
    layout="centered"
)

# Title
st.title("📩 Spam SMS Detection")
st.write("This application uses Machine Learning to classify SMS messages as **Spam** or **Legitimate (Ham)**.")

# Input box
message = st.text_area("Enter your SMS message:")

# Predict button
if st.button("Predict"):

    if message.strip() == "":
        st.warning("⚠ Please enter a message.")
    else:
        message_vector = vectorizer.transform([message])

        prediction = model.predict(message_vector)[0]
        probability = model.predict_proba(message_vector)

        if prediction == 1:
            st.error("🚨 Spam Message")
            st.write(f"**Confidence:** {probability[0][1]*100:.2f}%")
        else:
            st.success("✅ Legitimate (Ham) Message")
            st.write(f"**Confidence:** {probability[0][0]*100:.2f}%")

st.markdown("---")
st.caption("Developed by  Swathi | CodeSoft AI Internship")