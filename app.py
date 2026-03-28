import streamlit as st
import pickle

from src.predict import predict_email

# Load model and vectorizer
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

st.title("📧 Spam Email Classifier")

st.write("Enter an email message below:")

email = st.text_area("Email text")

if st.button("Predict"):
    if email.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, prob = predict_email(email, vectorizer, model)
        st.success(f"Prediction: {label}")
        st.write(f"Confidence: {prob:.2f}")