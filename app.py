import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array

# Load the model
model = tf.keras.models.load_model('cat_dog_model.keras')  # or .h5

# Title
st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image to find out if it's a **cat or dog**!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Prediction function using your code
def predict(img_path):
    img = load_img(img_path, target_size=(128, 128))   # RGB by default
    img = img_to_array(img) / 255.0                    # Normalize
    img = np.expand_dims(img, axis=0)                  # Reshape: (1, 128, 128, 3)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        return "Dog 🐶"
    else:
        return "Cat 🐱"

# Display image and prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save to temp file to use with load_img
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    result = predict("temp.jpg")
    st.markdown(f"### Prediction: **{result}**")
