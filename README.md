🌿 Plant Disease Detection

A web application to detect plant leaf diseases using a Convolutional Neural Network (CNN) model.

About the Project

This project leverages deep learning to detect common plant diseases from leaf images. Users can upload an image of a plant leaf, and the application predicts the disease with confidence scores.

The model is trained on a dataset of various plant leaf images and diseases including Tomato, Potato, Corn, and more.


Features

Upload a leaf image and get disease prediction

High accuracy using a CNN model

Simple, interactive web interface built with Streamlit

Supports multiple plant species and disease types


Tech Stack

Python – Programming language

TensorFlow / Keras – Deep learning framework

OpenCV / Pillow – Image processing

Streamlit – Web interface for deploying the model

Git & GitHub – Version control


Installation

Clone the repository:

git clone https://github.com/AKSHAYA1323/plant-disease-detection.git
cd plant-disease-detection


Create a virtual environment and activate it:

python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate  # Linux / Mac


Install the required packages:

pip install -r requirements.txt

Usage

Start the Streamlit app:

streamlit run src/app.py


Open the local URL provided in the terminal.

Upload a leaf image to get predictions with confidence scores.

Folder Structure

plant-disease-project/
│
├─ src/                  # Source code

│   ├─ app.py            # Streamlit app

│   ├─ inference.py      # Model inference code

│   ├─ model.py          # CNN model definition

│   ├─ preprocess.py     # Image preprocessing functions

│   ├─ split_data.py     # Data splitting script
│   └─ train.py          # Model training script
│
├─ notebooks/            # Jupyter notebooks
│   ├─ 01_explore.ipynb
│   ├─ 02_train.ipynb
│   └─ 03_inference.ipynb
│
├─ models/               # Saved trained models
│   └─ best_model.h5
│
├─ requirements.txt
├─ README.md
└─ .gitignore

Contributing

Fork the repository.

Create a new branch (git checkout -b feature/YourFeature).

Make your changes and commit (git commit -m "Add some feature").

Push to the branch (git push origin feature/YourFeature).

Create a Pull Request.

