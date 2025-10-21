import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import os

# --- IMPORTANT CHANGES ---
# 1. Using a relative path to point to the model
# 2. Make sure you have renamed your best model file to this name
MODEL_PATH = '../saved_models/garbage_classifier_best.h5'

# 3. Corrected the spelling of 'plastic'
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Use the newer, more efficient caching syntax for resources like models
@st.cache_resource
def load_our_model():
    """Load the trained Keras model."""
    try:
        # Check if the file exists before loading
        if not os.path.exists(MODEL_PATH):
            st.error(f"Error: Model file not found at path: {MODEL_PATH}")
            st.error("Please check the folder structure and filename.")
            return None
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess the image to fit the model's input."""
    # 4. Corrected the resize dimensions
    img_resized = cv2.resize(np.array(image), (224, 224))
    img_array = np.asarray(img_resized) / 255.0
    img_expanded = np.expand_dims(img_array, axis=0)
    return img_expanded

# --- Application Interface ---
st.set_page_config(page_title="Garbage Classifier")
st.title("AI Garbage Classifier")
st.write("Upload an image of a piece of trash, and the AI will classify it.")

# Load the model
model = load_our_model()

# Only show the file uploader if the model loaded successfully
if model is not None:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open and display the image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess and predict
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(prediction) * 100

        # Display the result
        st.success(f"**Prediction:** {predicted_class_name.capitalize()} ({confidence:.2f}%)")

        # Display additional information
        RECYCLING_INFO = {
            "cardboard": "Recyclable! Please flatten and place in the paper bin.",
            "glass": "Recyclable! Please clean before placing in the glass bin.",
            "metal": "Recyclable! This is metal, please dispose of it in the designated place.",
            "paper": "Recyclable! Please place in the paper bin.",
            "plastic": "Recyclable! Please clean and place in the plastic bin.",
            "trash": "Not recyclable. Please dispose of in the general waste bin."
        }
        info = RECYCLING_INFO.get(predicted_class_name, "No information available for this category.")
        st.info(info)
else:
    st.warning("Unable to load the model. Please check the terminal for detailed error messages.")

st.sidebar.header("About")
st.sidebar.info(
    "This is a web application that uses a deep learning model (Transfer Learning with EfficientNetB0) "
    "to classify waste into different categories. "
    "The model was trained on Google Colab and is deployed with Streamlit."
)