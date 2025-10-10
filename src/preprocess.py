import numpy as np
import cv2
import os
import random
from glob import glob
from sklearn.datasets import fetch_lfw_people

def load_faces_in_things(path="data/faces_in_things", size=(128,128)):
    files = glob(os.path.join(path, "*.jpg"))
    imgs = []
    for f in files:
        img = cv2.imread(f)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, size)
            imgs.append(img / 255.0)
    return np.array(imgs)

def load_lfw_faces(size=(128,128)):
    lfw = fetch_lfw_people(color=True, resize=0.5)
    imgs = [cv2.resize(img, size) for img in lfw.images]
    return np.array(imgs) / 255.0

def make_pairs(face_imgs, nonface_imgs, n_pairs=5000):
    pairs, labels = [], []
    for _ in range(n_pairs):
        if random.random() > 0.5:
            if random.random() > 0.5:
                a, b = random.sample(list(face_imgs), 2)
            else:
                a, b = random.sample(list(nonface_imgs), 2)
            label = 1
        else:
            a = random.choice(face_imgs)
            b = random.choice(nonface_imgs)
            label = 0
        pairs.append((a, b))
        labels.append(label)
    return np.array(pairs), np.array(labels)
