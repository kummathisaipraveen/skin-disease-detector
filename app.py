import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
MODEL_PATH = "model.h5"  # Update with your actual model path
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
CLASS_NAMES = ['Acne', 'Eczema', 'Psoriasis', 'Rosacea', 'Ringworm', 
               'Melanoma', 'Dermatitis', 'Urticaria', 'Vitiligo', 'Lupus']

# Define precautions based on disease
PRECAUTIONS = {
    "Acne": "Wash your face twice daily, avoid oily foods.",
    "Eczema": "Use mild soap, apply moisturizer regularly.",
    "Psoriasis": "Use medicated creams, avoid scratching.",
    "Rosacea": "Avoid spicy foods, protect your skin from the sun.",
    "Ringworm": "Keep skin dry, avoid sharing personal items.",
    "Melanoma": "Use sunscreen, avoid excessive sun exposure.",
    "Dermatitis": "Avoid allergens, use prescribed creams.",
    "Urticaria": "Avoid allergens, take antihistamines if needed.",
    "Vitiligo": "Use sunscreen, consult a dermatologist.",
    "Lupus": "Stay out of the sun, take prescribed medication."
}

# Function to preprocess image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to match model input
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():  # Corrected function signature
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the image file
    image = Image.open(io.BytesIO(file.read()))
    
    # Preprocess and make prediction
    img = preprocess_image(image)
    prediction = model.predict(img)
    
    # Get the predicted disease
    predicted_label = CLASS_NAMES[np.argmax(prediction)]
    precautions = PRECAUTIONS.get(predicted_label, "No specific precautions available.")

    # Return JSON response
    return jsonify({
        "Predicted Disease": predicted_label,
        "Precautions": precautions
    })

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
