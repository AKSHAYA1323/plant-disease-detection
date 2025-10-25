import os
from PIL import Image

def resize_images(source_folder, size=(224, 224)):
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(size)
                    img.save(img_path)
                except Exception as e:
                    print(f"❌ Error processing {img_path}: {e}")

if __name__ == "__main__":
    resize_images("../data/raw")
    print("✅ All images resized successfully!")
