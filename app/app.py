Ô chứa mã <F4EmY2FcJEeD>
# %% [code]
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import os
import numpy as np
import cv2

# Replace with the actual path to your model file
MODEL_PATH = '/content/save_models/garbage_classifier_model.h5' # Placeholder path
CLASS_NAME = ['cardboard', 'glass', 'metal', 'paper', 'platstic', 'trash']


@st.cache(allow_output_mutation=True)
def load_our_model():
  try:
    model = load_model(MODEL_PATH)
    return model
  except Exception as e:
    st.error(f"Lỗi khi tải mô hình...{e}")
    return None

    #Unify data preprocessing.
def preprocess_image(image):
  img_resize = cv2.resize(np.array(image), (224, 244))
  img_array = np.asarray(img_resize) / 255.0
  img_expanded = np.expand_dims(img_array, axis=0)
  return img_expanded


  st.set_page_config(page_title="Garbage Classifier", page_icon=":recycle:")
st.title("Garbage Classifier")
st.write("upload an image of a piece of trash, and AI will classify it.")

#download model
model = load_our_model()
if model is not None:
  uploaded_file = st.file_uploader("Chọn một hình ảnh...", type=("jpg", "jpeg", "png"))
  if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Hình ảnh tải lên", use_column_width=True)
    preprocessed_image = preprocess_image(image)

    #prediction
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = CLASS_NAME[predicted_class_index]
    confidence = np.max(prediction) * 100
    st.write(f"Kết quả dự đoán: {predicted_class_name.capitalize()}")
    st.info(f"Độ chính xác: {confidence:.2f}%")

else:
  st.error("Unable to load model. Please check the model path and file again.")

st.sidebar.header("About the project")
st.sidebar.info(
    "This is a web application that uses a deep learning model (Transfer Learning with MobileNetV2) "
  "to classify waste into different types. "
  "The model is trained on Google Colab and deployed with Streamlit."
  )