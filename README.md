# SIAMESE NEURAL NETWORK FOR PAREIDOLIA MITIGATION

Developed a Siamese Neural Network (SNN) in Keras/TensorFlow with shared-weight convolutional architecture to distinguish genuine human faces from pareidolic face-like patterns by learning similarity embeddings via contrastive loss and Euclidean distance on paired inputs from the LFW and Faces in Things datasets; achieved 99.6% accuracy, 0.999 AUC, and F1-scores > 0.995 across 42 ablation experiments, with robust generalization to unseen pareidolia categories using as little as 0.5% of non-face training data. Research paper document included.

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
