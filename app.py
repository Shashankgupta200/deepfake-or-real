
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image

# # Load the trained model
# MODEL_PATH = "deepfake_detector_model.h5"
# model = tf.keras.models.load_model(MODEL_PATH)

# # Define image preprocessing function
# def preprocess_image(image):
#     image = image.resize((256, 256))  # Resize to match model input
#     image = np.array(image) / 255.0   # Normalize
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image

# # Streamlit UI
# st.title("Deepfake Image Detector")
# st.write("Upload an image to check if it is Real or Deepfake.")

# # File uploader
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Process image and make prediction
#     processed_image = preprocess_image(image)
#     prediction = model.predict(processed_image)[0][0]

#     # Display result
#     result = "Deepfake" if prediction > 0.5 else "Real"
#     confidence = prediction if prediction > 0.5 else 1 - prediction

#     st.subheader(f"Prediction: **{result}**")
#     st.write(f"Confidence: {confidence:.2%}")



# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from PIL import Image

# # Constants
# IMG_HEIGHT = 256
# IMG_WIDTH = 256

# # Load trained model
# @st.cache_resource()
# def load_deepfake_model():
#     return load_model("deepfake_detector_model.h5")

# model = load_deepfake_model()

# # Function to predict image
# def predict_image(img):
#     img = img.resize((IMG_HEIGHT, IMG_WIDTH))
#     img_array = image.img_to_array(img) / 255.0  # Normalize
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     prediction = model.predict(img_array)[0][0]
#     return "Real" if prediction > 0.5 else "Deepfake"

# # Streamlit UI
# st.title("Deepfake Image Detector")
# st.write("Upload an image to check if it's real or deepfake.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image_pil = Image.open(uploaded_file)
#     st.image(image_pil, caption="Uploaded Image", use_column_width=True)
    
#     # Predict and display result
#     result = predict_image(image_pil)
#     st.write(f"**Prediction:** {result}")



import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
MODEL_PATH = "deepfake_detector_model.h5"
model = load_model(MODEL_PATH)

# Define image dimensions
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Function to predict the image
def predict_image(img, model):
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))  # Resize image
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)[0][0]  # Get prediction score

    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
    label = "Real" if prediction > 0.5 else "Deepfake"

    return label, confidence

# Streamlit UI
st.title("Deepfake Image Detection")
st.write("Upload an image to check if it's real or a deepfake.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image_display = image.load_img(uploaded_file)
    st.image(image_display, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        label, confidence = predict_image(image_display, model)
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}%")
