import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Define model path (Update this if your model is stored in a different location)
MODEL_PATH = r"skin_disease_detector.h5"  # Ensure this path is correct
model = tf.keras.models.load_model(MODEL_PATH)

# Define disease names (Ensure these match your model's output classes)
diseases = [
    "Acne", "Eczema", "Psoriasis", "Rosacea", "Ringworm",
    "Melanoma", "Vitiligo", "Impetigo", "Urticaria", "Lupus"
]

# Define precautions for each disease
PRECAUTIONS = {
    "Acne": [
        "Wash your face twice a day with mild cleanser.",
        "Avoid touching your face frequently.",
        "Use non-comedogenic moisturizers and sunscreen.",
        "Stay hydrated and eat a balanced diet."
    ],
    "Eczema": [
        "Moisturize skin regularly with fragrance-free lotions.",
        "Avoid triggers like harsh soaps and detergents.",
        "Wear soft cotton clothing to reduce irritation.",
        "Use mild, hypoallergenic soaps for bathing."
    ],
    "Psoriasis": [
        "Keep skin moisturized to prevent dryness.",
        "Avoid smoking and excessive alcohol consumption.",
        "Manage stress through meditation or yoga.",
        "Use medicated shampoos if scalp psoriasis is present."
    ],
    "Rosacea": [
        "Avoid hot beverages and spicy foods.",
        "Use gentle skincare products with no alcohol or fragrances.",
        "Protect your face from the sun with sunscreen.",
        "Reduce stress to prevent flare-ups."
    ],
    "Ringworm": [
        "Keep affected area clean and dry.",
        "Avoid sharing personal items like towels or clothing.",
        "Use antifungal creams or powders as prescribed.",
        "Wash hands after touching infected areas."
    ],
    "Melanoma": [
        "Avoid prolonged sun exposure and use sunscreen.",
        "Check skin regularly for any new or changing moles.",
        "Wear protective clothing when in the sun.",
        "Consult a dermatologist for any suspicious growths."
    ],
    "Dermatitis": [
        "Use fragrance-free and hypoallergenic skincare products.",
        "Avoid excessive scratching to prevent infections.",
        "Apply cool compresses to soothe itching.",
        "Use prescribed corticosteroid creams if needed."
    ],
    "Urticaria": [
        "Avoid known allergens and triggers.",
        "Apply cold compresses to reduce itching and swelling.",
        "Use antihistamines as directed by a doctor.",
        "Wear loose-fitting, soft clothing."
    ],
    "Vitiligo": [
        "Protect skin from excessive sun exposure.",
        "Use sunscreen with high SPF regularly.",
        "Eat a healthy diet rich in antioxidants.",
        "Consult a dermatologist for phototherapy options."
    ],
    "Lupus": [
        "Avoid excessive sun exposure and use sunscreen.",
        "Stay hydrated and eat a balanced diet.",
        "Manage stress levels through relaxation techniques.",
        "Consult a doctor for appropriate medications."
    ]
}

# Ensure upload directory exists
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowable image file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Ensure this matches your model's input size
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Preprocess image and predict
    img = preprocess_image(file_path)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    disease_name = diseases[predicted_class]
    precaution = precautions.get(disease_name, "No specific precautions available.")

    result = {
        "disease": disease_name,
        "precautions": precaution
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
