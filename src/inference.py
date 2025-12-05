import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ✅ Absolute path fix for local + Streamlit cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_model.h5")

model = load_model(MODEL_PATH)

# Labels must match the training classes
labels = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Healthy'
]

def predict(input_data):
    """
    Predict the disease from a PIL image or a file path.
    """
    if isinstance(input_data, str):
        img = Image.open(input_data).convert('RGB')
    elif isinstance(input_data, Image.Image):
        img = input_data
    else:
        raise ValueError("Input must be a PIL Image or a file path.")

    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    index = np.argmax(preds)

    return labels[index], float(preds[index])
