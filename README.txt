# Pareidolia Siamese Neural Network

This project distinguishes real human faces (LFW) from face-like patterns (Faces in Things dataset) using a Siamese Neural Network built in TensorFlow/Keras.

## Setup
```bash
git clone https://github.com/wll1521/pareidolia-snn.git
cd pareidolia-snn
pip install -r requirements.txt
```

## Download only once if you don't already have the faces in things dataset:
```bash
python -m src.download_faces_in_things
```

## Run:
```bash
python -m src.train_snn
```
