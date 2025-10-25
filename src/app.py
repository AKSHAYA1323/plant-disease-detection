import streamlit as st
from PIL import Image
from inference import predict

st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to detect its disease using a CNN model.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Display image
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Predicting...")

    # Predict directly from PIL image
    label, confidence = predict(image)
    st.success(f"Prediction: **{label}** ({confidence*100:.2f}%)")
