import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model

def train_model(data_dir, epochs=10, batch_size=32):
    img_size = (224, 224)
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    print(f"📁 Loading data from {data_dir} ...")

    train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20,
                                   zoom_range=0.2, horizontal_flip=True)
    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(train_dir, target_size=img_size,
                                               batch_size=batch_size, class_mode='categorical')
    val_data = val_gen.flow_from_directory(val_dir, target_size=img_size,
                                           batch_size=batch_size, class_mode='categorical')

    num_classes = len(train_data.class_indices)
    model = build_model(num_classes=num_classes, input_shape=img_size + (3,))

    print("🚀 Starting training...")
    history = model.fit(train_data, validation_data=val_data, epochs=epochs)

    os.makedirs("../models", exist_ok=True)
    model.save("../models/best_model.h5")
    print("✅ Model saved at ../models/best_model.h5")

    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/processed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    train_model(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
