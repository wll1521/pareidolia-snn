import os
import re
import random
from glob import glob
from typing import Tuple, Dict, Any, List
import cv2
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


####### Preprocessing #######
# Can change image size for faster or more accurate results
# LFW default size is 250 so I just resized facesInthings to 250 as well (originally 500)
# 250 requires around 12 gigs of RAM to store LFW dataset

img_size = 64

filter_cat_ind = "num_boxes"         # Only uses photos with this column value
filter_with_ind = 1                  # Value in filter_cat_ind
filtered_cat = "Accident or design?" # Category column to filter on
holdout_category = "Design"          # Hold-out pareidolia category (test only)

data_fraction = 1.0      # 0.5 uses half of available training images (around 1,900)
n_pairs_train = 9000     # training pairs
n_pairs_test = 3000      # test pairs



def load_faces_in_things(path: str = "data/faces_in_things",
                         size: Tuple[int, int] = (img_size, img_size)) -> np.ndarray:
    files = glob(os.path.join(path, "*.jpg"))
    files = sorted(files, key=lambda f: int(re.search(r"img_(\d+)", os.path.basename(f)).group(1)))
    imgs: List[np.ndarray] = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        imgs.append(img.astype(np.float32) / 255.0)
    return np.array(imgs)


def load_lfw_faces(size: Tuple[int, int] = (img_size, img_size)) -> np.ndarray:
    lfw = fetch_lfw_people(color=True, resize=1.0)
    imgs: List[np.ndarray] = []
    for img in lfw.images:
        img = img.astype(np.float32)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        imgs.append(img)
    return np.stack(imgs, axis=0)


def load_from_meta(df: pd.DataFrame,
                   size: Tuple[int, int] = (img_size, img_size),
                   root: str = "data/faces_in_things") -> np.ndarray:
    out: List[np.ndarray] = []
    for idx in df.index:
        img_path = os.path.join(root, f"img_{idx}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        out.append(img.astype(np.float32) / 255.0)
    return np.array(out)


# Create pairs
def make_pairs_strict(face_imgs: np.ndarray,
                      nonface_imgs: np.ndarray,
                      n_pairs: int,
                      nonface_categories: List[str] | None = None):
    pairs, labels, types = [], [], []
    cat_labels = [] if nonface_categories is not None else None

    n_each = n_pairs // 3

    # Face–Face (positive / 1)
    for _ in range(n_each):
        idx_a, idx_b = random.sample(range(len(face_imgs)), 2)
        a, b = face_imgs[idx_a], face_imgs[idx_b]
        pairs.append((a, b))
        labels.append(1)
        types.append("Face-Face")
        if cat_labels is not None:
            cat_labels.append(("Face", "Face"))

    # NonFace–NonFace (positive / 1)
    for _ in range(n_each):
        idx_a, idx_b = random.sample(range(len(nonface_imgs)), 2)
        a, b = nonface_imgs[idx_a], nonface_imgs[idx_b]
        pairs.append((a, b))
        labels.append(1)
        types.append("NonFace-NonFace")
        if cat_labels is not None:
            cat_labels.append((nonface_categories[idx_a], nonface_categories[idx_b]))

    # Face–NonFace (negative / 0)
    for _ in range(n_each):
        idx_a = random.randrange(len(face_imgs))
        idx_b = random.randrange(len(nonface_imgs))
        a, b = face_imgs[idx_a], nonface_imgs[idx_b]
        pairs.append((a, b))
        labels.append(0)
        types.append("Face-NonFace")
        if cat_labels is not None:
            cat_labels.append(("Face", nonface_categories[idx_b]))

    # Shuffle
    if cat_labels is not None:
        combined = list(zip(pairs, labels, types, cat_labels))
        random.shuffle(combined)
        pairs, labels, types, cat_labels = zip(*combined)
        return np.array(pairs), np.array(labels), np.array(types), np.array(cat_labels)
    else:
        combined = list(zip(pairs, labels, types))
        random.shuffle(combined)
        pairs, labels, types = zip(*combined)
        return np.array(pairs), np.array(labels), np.array(types)


def split_pairs(pairs: np.ndarray):
    a = np.array([x[0] for x in pairs])
    b = np.array([x[1] for x in pairs])
    return [a, b]


def show_random_pairs(pairs: np.ndarray,
                      labels: np.ndarray,
                      types: np.ndarray,
                      n: int = 5) -> None:
    plt.figure(figsize=(10, 2 * n))
    for i in range(n):
        idx = random.randint(0, len(pairs) - 1)
        a, b = pairs[idx]
        label = labels[idx]
        pair_type = types[idx]

        plt.subplot(n, 2, 2 * i + 1)
        plt.imshow(a)
        plt.axis("off")
        plt.title(f"{pair_type} | Label={label}")

        plt.subplot(n, 2, 2 * i + 2)
        plt.imshow(b)
        plt.axis("off")
        plt.title(f"{pair_type} | Label={label}")

    plt.tight_layout()
    plt.show()


# Dataset Builder
def build_train_test_pairs(
    metadata_path: str = "data/metadata.csv",
    seed: int = 1,
) -> Dict[str, Any]:

    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    metadata = pd.read_csv(metadata_path)

    print("Loading datasets…")
    faces = load_lfw_faces(size=(img_size, img_size))
    nonfaces = load_faces_in_things(size=(img_size, img_size))
    print(f"Faces: {faces.shape}, Nonfaces: {nonfaces.shape}")

    # 1. Filters the full non‑face set by num_boxes or whatever other criteria was selected at the top
    valid_idx = metadata[metadata[filter_cat_ind] == filter_with_ind].index.tolist()
    nonfaces_filtered = nonfaces[valid_idx]
    categories_filtered = [metadata.loc[i, filtered_cat] for i in valid_idx]

    # 2. Split faces and non‑faces into train/test (apply category holdout on non‑faces if desired from the top)
    faces_train, faces_test = train_test_split(faces, test_size=0.2, random_state=seed)

    # 3. Hold out a specific pareidolia category from the nonface set for evaluation only
    train_indices = [i for i, c in enumerate(categories_filtered) if c != holdout_category]
    test_indices = [i for i, c in enumerate(categories_filtered) if c == holdout_category]

    nonfaces_train_full = nonfaces_filtered[train_indices]
    nonfaces_test_full = nonfaces_filtered[test_indices]
    nonface_categories_train = [categories_filtered[i] for i in train_indices]
    nonface_categories_test = [categories_filtered[i] for i in test_indices]

    # 4. Apply data_fraction to training sets
    if data_fraction < 1.0:
        face_train_idx = rng.choice(len(faces_train), int(len(faces_train) * data_fraction), replace=False)
        nonface_train_idx = rng.choice(
            len(nonfaces_train_full),
            int(len(nonfaces_train_full) * data_fraction),
            replace=False,
        )
        faces_train_subset = faces_train[face_train_idx]
        nonfaces_train_subset = nonfaces_train_full[nonface_train_idx]
        nonface_categories_train_subset = [nonface_categories_train[i] for i in nonface_train_idx]
    else:
        faces_train_subset = faces_train
        nonfaces_train_subset = nonfaces_train_full
        nonface_categories_train_subset = nonface_categories_train

    # 5. Build train/test pairs
    train_pairs, y_train, types_train, cat_train = make_pairs_strict(
        faces_train_subset,
        nonfaces_train_subset,
        n_pairs_train,
        nonface_categories=nonface_categories_train_subset,
    )
    test_pairs, y_test, types_test, cat_test = make_pairs_strict(
        faces_test,
        nonfaces_test_full,
        n_pairs_test,
        nonface_categories=nonface_categories_test,
    )

    print("Training pairs:", len(train_pairs))
    print("Testing pairs:", len(test_pairs))

    return {
        "metadata": metadata,
        "faces_train": faces_train_subset,
        "faces_test": faces_test,
        "nonfaces_train": nonfaces_train_subset,
        "nonfaces_test": nonfaces_test_full,
        "train_pairs": train_pairs,
        "y_train": y_train,
        "types_train": types_train,
        "cat_train": cat_train,
        "test_pairs": test_pairs,
        "y_test": y_test,
        "types_test": types_test,
        "cat_test": cat_test,
        "filter_cat_ind": filter_cat_ind,
        "filtered_cat": filtered_cat,
        "holdout_category": holdout_category,
    }
