from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the trained deep learning model
model = tf.keras.models.load_model("model.h5")  # Change "model.h5" to your actual model file

# Define a function to preprocess images
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))  # Read the image
    processed_image = preprocess_image(image)   # Preprocess the image

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the class with the highest probability

    return jsonify({"prediction": int(predicted_class)})

# Run the API
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
