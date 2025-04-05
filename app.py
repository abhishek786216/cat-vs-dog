import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="cat_dog_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Title
st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image to find out if it's a **cat or dog**!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Prediction function using TFLite model
def predict(image: Image.Image):
    # Resize and preprocess
    img = image.resize((128, 128))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    interpreter.invoke()

    # Get prediction result
    prediction = interpreter.get_tensor(output_details[0]['index'])

    if prediction[0][0] > 0.5:
        return "Dog 🐶"
    else:
        return "Cat 🐱"

# Display image and prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    result = predict(image)
    st.markdown(f"### Prediction: **{result}**")
