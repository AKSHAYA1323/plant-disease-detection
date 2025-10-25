import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    classes = os.listdir(source_dir)
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        images = os.listdir(cls_path)
        train, temp = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
        val, test = train_test_split(temp, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42)

        for subset, subset_data in zip(["train", "val", "test"], [train, val, test]):
            subset_dir = os.path.join(dest_dir, subset, cls)
            os.makedirs(subset_dir, exist_ok=True)
            for img in subset_data:
                shutil.copy(os.path.join(cls_path, img), os.path.join(subset_dir, img))

if __name__ == "__main__":
    split_data("../data/raw", "../data/processed")
    print("✅ Data split into train/val/test successfully!")
