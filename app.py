# Import required libraries
import streamlit as st  
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Page setup
st.set_page_config(
    page_title="Tomato Disease Detection",
    page_icon="🍅",
    initial_sidebar_state="auto"
)

# Hide Streamlit footer and menu
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Helper functions ---
def prediction_cls(prediction, class_names):
    return class_names[np.argmax(prediction)]

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...] / 255.0  # Normalize if required
    prediction = model.predict(img_reshape)
    return prediction

# Sidebar layout
with st.sidebar:
    st.title("TomatoCare 🌱")
    st.subheader("Detect tomato leaf diseases and get instant remedies.")

# Page title
st.title("🍅 Tomato Disease Detection")
st.write("Upload a tomato leaf image to detect if it's healthy or has Early/Late Blight.")

# File uploader
file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "png"])

# Load your model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("model_resnet50.h5")  # <-- your model file here
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Define your 3 class names
class_names = ['Early Blight', 'Late Blight', 'Healthy']

# Prediction workflow
if file is None:
    st.info("Please upload an image to proceed.")
else:
    image = Image.open(file)
    st.image(image, use_container_width=True, caption="Uploaded Tomato Leaf Image")

    if model:
        with st.spinner("Predicting..."):
            try:
                predictions = import_and_predict(image, model)
                predicted_class = prediction_cls(predictions, class_names)
                confidence = round(100 * np.max(predictions), 2)  # Confidence from the prediction
                st.sidebar.success(f"✅ Confidence: {confidence}%")
                st.subheader(f"🦠 Detected: {predicted_class}")

                # Remedies
                if predicted_class == 'Healthy':
                    st.success("This tomato leaf looks healthy. No signs of disease detected.")
                    st.balloons()

                elif predicted_class == 'Early Blight':
                    st.markdown("### 🩺 Remedy for Early Blight")
                    st.info("Apply fungicides like chlorothalonil or mancozeb. Remove infected leaves. Use resistant plant varieties. Avoid overhead watering and rotate crops regularly.")

                elif predicted_class == 'Late Blight':
                    st.markdown("### 🩺 Remedy for Late Blight")
                    st.info("Use copper-based fungicides immediately. Destroy affected plants. Avoid moisture on leaves and improve air circulation around plants.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.error("Model failed to load.")
