import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "deepfake_detector_model.h5"
model = load_model(MODEL_PATH)

IMG_HEIGHT = 256
IMG_WIDTH = 256

def predict_image(img, model):
    img = img.resize((IMG_HEIGHT, IMG_WIDTH)) 
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 

    prediction = model.predict(img_array)[0][0] 

    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
    label = "Real" if prediction > 0.5 else "Deepfake"

    return label, confidence

st.title("Deepfake Image Detection")
st.write("Upload an image to check if it's real or a deepfake.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_display = image.load_img(uploaded_file)
    st.image(image_display, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        label, confidence = predict_image(image_display, model)
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}%")
