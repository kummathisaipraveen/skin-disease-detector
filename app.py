import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# Load the pre-trained model
MODEL_PATH = "skin_disease_detector.h5"  # Ensure this path is correct
model = tf.keras.models.load_model(MODEL_PATH)

# Define disease names
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

# Streamlit app layout
st.title("Skin Disease Detection and Precaution Guide")
st.write("Upload an image to predict the skin disease and get recommended precautions.")

# Image upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))  # Ensure this matches your model's input size
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Predict function
def predict_image(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    disease_name = diseases[predicted_class]
    precautions = PRECAUTIONS.get(disease_name, "No specific precautions available.")
    return disease_name, precautions

# Display the uploaded image and prediction results
if uploaded_file is not None:
    # Show the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict and display results
    disease, precautions = predict_image(uploaded_file)
    st.subheader(f"Predicted Disease: {disease}")
    st.write("**Precautions:**")
    for precaution in precautions:
        st.write(f"- {precaution}")

