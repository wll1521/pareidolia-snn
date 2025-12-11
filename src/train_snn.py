import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from .preprocess import img_size, data_fraction, build_train_test_pairs, split_pairs, show_random_pairs
from .model_snn import build_snn
from .evaluation import evaluate_model
from .logger import log_experiment


######## Training #######
# Main thing to change here is epochs and batch sizes
# Batch size controls how many training samples are processed before the model updates its weights (more mem required)
# If too small loss will fluctuate, too large converges on local minimum
# Can also change early stopping patience. I keep it on 5 with a max of 40. The model tends to converge prior to 20 epochs

def set_seeds(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    seed1 = 0
    epoch_num = 1
    batch_num = 32

    set_seeds(seed1)

    # Builds training & test data
    data = build_train_test_pairs(seed=seed1)
    metadata = data["metadata"]
    train_pairs = data["train_pairs"]
    y_train = data["y_train"]
    test_pairs = data["test_pairs"]
    y_test = data["y_test"]
    types_test = data["types_test"]
    cat_test = data["cat_test"]
    faces_test = data["faces_test"]
    filter_cat_ind = data["filter_cat_ind"]
    filtered_cat = data["filtered_cat"]
    holdout_category = data["holdout_category"]

    print("Showing training pair examples:")
    show_random_pairs(train_pairs, y_train, data["types_train"], n=5)

    print("Showing testing pair examples:")
    show_random_pairs(test_pairs, y_test, data["types_test"], n=5)

    # Builds model
    input_shape = (img_size, img_size, 3)
    model = build_snn(input_shape=input_shape)
    model.summary()

    print("Training model with contrastive loss…")

    # Validation split on test for contrastive lost function and threshold
    train_pairs_split, val_pairs, y_train_split, y_val = train_test_split(
        train_pairs, y_train, test_size=0.2, random_state=seed1, stratify=y_train
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    start_time = time.time()
    history = model.fit(
        split_pairs(train_pairs_split),
        y_train_split,
        validation_data=(split_pairs(val_pairs), y_val),
        epochs=epoch_num,
        batch_size=batch_num,
        callbacks=[early_stop],
        verbose=2,
    )
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")

    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    print("Best epoch (early stopping restored from epoch):", best_epoch)

    metrics = evaluate_model(
        model=model,
        val_pairs=val_pairs,
        y_val=y_val,
        test_pairs=test_pairs,
        y_test=y_test,
        types_test=types_test,
        cat_test=cat_test,
        metadata=metadata,
        filter_cat_ind=filter_cat_ind,
        filtered_cat=filtered_cat,
        holdout_category=holdout_category,
        faces_test=faces_test,
    )

    ts = datetime.now().isoformat()

    log_row = {
        "timestamp": ts,
        "seed": seed1,
        "data_fraction": data_fraction,
        "batch_num": batch_num,
        "img_size": img_size,
        "num_pairs_train": len(train_pairs),
        "num_pairs_test": len(test_pairs),
        "best_epoch": best_epoch,
        "train_time(s)": training_time,
    }

    log_row.update(metrics)
    log_experiment("experiment_log.csv", log_row)

if __name__ == "__main__":
    main()
