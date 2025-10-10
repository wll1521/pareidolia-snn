# Pareidolia Siamese Neural Network (CSC 526)

This project distinguishes real human faces (LFW) from face-like patterns (Faces in Things dataset) using a Siamese Neural Network built in TensorFlow/Keras.

## Setup

```bash
git clone https://github.com/<your-username>/pareidolia-snn.git
cd pareidolia-snn
pip install -r requirements.txt


## Run 
## only once if you haven't
python src/download_faces_in_things.py

## creates model and runs testing
python src/train_snn.py
