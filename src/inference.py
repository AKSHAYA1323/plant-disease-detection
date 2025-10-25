import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
MODEL_PATH = "../models/best_model.h5"
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
    
    Args:
        input_data: PIL Image or path to image file
    Returns:
        label (str), confidence (float)
    """
    # If input is a file path, open it with PIL
    if isinstance(input_data, str):
        img = Image.open(input_data).convert('RGB')
    elif isinstance(input_data, Image.Image):
        img = input_data
    else:
        raise ValueError("Input must be a PIL Image or a file path.")

    # Resize and convert to array
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)[0]
    index = np.argmax(preds)
    return labels[index], float(preds[index])
