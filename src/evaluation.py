import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .preprocess import split_pairs, load_from_meta, make_pairs_strict
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

def evaluate_model(
    model,
    val_pairs,
    y_val,
    test_pairs,
    y_test,
    types_test,
    cat_test,
    metadata,
    filter_cat_ind: str,
    filtered_cat: str,
    holdout_category: str,
    faces_test,
    category_sep: str = "Person or creature?",
):

    print("Evaluating model…")

    # Threshold selection on validation
    val_dist = model.predict(split_pairs(val_pairs)).ravel()
    val_sim = -val_dist

    fpr_val, tpr_val, thresh_val = roc_curve(y_val, val_sim)
    best_idx = np.argmax(tpr_val - fpr_val)
    best_thresh_sim = thresh_val[best_idx]
    best_thresh_dist = float(-best_thresh_sim)

    # Test distances
    test_start = time.time()
    test_dist = model.predict(split_pairs(test_pairs)).ravel()
    test_eval_time = time.time() - test_start
    print(f"Test evaluation time: {test_eval_time:.4f} seconds")
    test_sim = -test_dist

    # Binary predictions using similarity threshold
    pred_labels = (test_sim > best_thresh_sim).astype("int32")

    # Recall, Precision and F1 score for classes
    precision_0, precision_1 = precision_score(y_test, pred_labels, average=None)
    recall_0, recall_1 = recall_score(y_test, pred_labels, average=None)
    f1_0, f1_1 = f1_score(y_test, pred_labels, average=None)

    precision_macro = precision_score(y_test, pred_labels, average="macro")
    recall_macro = recall_score(y_test, pred_labels, average="macro")
    f1_macro = f1_score(y_test, pred_labels, average="macro")

    precision_weighted = precision_score(y_test, pred_labels, average="weighted")
    recall_weighted = recall_score(y_test, pred_labels, average="weighted")
    f1_weighted = f1_score(y_test, pred_labels, average="weighted")

    # ROC & AUC on test
    fpr_test, tpr_test, _ = roc_curve(y_test, test_sim)
    roc_auc = auc(fpr_test, tpr_test)

    accuracy = float(np.mean(pred_labels == y_test))
    cm = confusion_matrix(y_test, pred_labels)
    tn, fp, fn, tp = cm.ravel()

    print("Test Accuracy @ val-threshold:", accuracy)
    print("Confusion Matrix:\n", cm)
    print(f"True Positives : {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives : {tn}")
    print(f"False Negatives: {fn}")
    print(f"Optimal distance threshold (from validation): {best_thresh_dist:.6f}")
    print(f"Test Accuracy @ val-threshold: {accuracy * 100:.2f}%")
    print(f"ROC-AUC: {roc_auc:.3f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ROC curve plot
    plt.figure(figsize=(6, 5))
    plt.plot(fpr_test, tpr_test, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    roc_path = RESULTS_DIR / f"roc_curve_{ts}.pdf"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.show()

    # Distance distributions plot
    plt.figure(figsize=(6, 4))
    plt.hist(test_dist[y_test == 1], bins=30, alpha=0.6, label="Similar pairs")
    plt.hist(test_dist[y_test == 0], bins=30, alpha=0.6, label="Dissimilar pairs")
    plt.axvline(best_thresh_dist, color="red", linestyle="--",
                label=f"Threshold={best_thresh_dist:.3f}")
    plt.xlabel("Distance (lower = more similar)")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Predicted Distance Distributions")
    dist_path = RESULTS_DIR / f"distance_distributions_{ts}.pdf"
    plt.savefig(dist_path, bbox_inches="tight")
    plt.show()

    # # Per-subcategory AUC/accuracy for hold-out category
    category_results: dict[str, dict[str, int | float]] = {}
    for i, pair_type in enumerate(types_test):
        cat_a, cat_b = cat_test[i]
        if pair_type == "Face-NonFace":
            cats = [cat_b]
        elif pair_type == "NonFace-NonFace":
            cats = [cat_a, cat_b]
        else:
            continue

        for cat in cats:
            if pd.isna(cat):
                continue
            if cat not in category_results:
                category_results[cat] = {"total": 0, "correct": 0}
            category_results[cat]["total"] += 1
            if pred_labels[i] == y_test[i]:
                category_results[cat]["correct"] += 1

    print("Per-category accuracy:")
    for cat, stats in category_results.items():
        acc_cat = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f" - {cat}: {acc_cat * 100:.2f}% ({stats['correct']}/{stats['total']})")

    total_correct = sum(s["correct"] for s in category_results.values())
    total_samples = sum(s["total"] for s in category_results.values())
    holdout_accuracy = (total_correct / total_samples) if total_samples > 0 else 0.0

    meta_test = metadata[
        (metadata[filter_cat_ind] == 1) &
        (metadata[filtered_cat] == holdout_category)
    ].copy()

    df_cat_rows = []
    cats = metadata[category_sep].dropna().unique().tolist()

    for cat in cats:
        sub = meta_test.loc[meta_test[category_sep] == cat]
        if sub.empty:
            continue
        nf = load_from_meta(sub)
        if len(nf) < 30:
            continue

        cat_pairs, cat_labels, _types, _ = make_pairs_strict(
            faces_test, nf, n_pairs=min(1000, len(nf) * 2),
            nonface_categories=["holdout"] * len(nf),
        )

        p = model.predict(split_pairs(cat_pairs), verbose=0).ravel()
        s = -p
        preds_binary = (s > best_thresh_sim).astype(int)
        acc_val = float(np.mean(preds_binary == cat_labels))

        fpr_c, tpr_c, _ = roc_curve(cat_labels, s)
        auc_val = auc(fpr_c, tpr_c)

        df_cat_rows.append((cat, len(nf), auc_val, acc_val))

    if df_cat_rows:
        df_cat = pd.DataFrame(
            df_cat_rows, columns=["Category", "Samples", "AUC", "Accuracy"]
        ).sort_values("AUC")
        print(df_cat)
        csv_path = RESULTS_DIR / f"per_category_{ts}.csv"
        df_cat.to_csv(csv_path, index=False)

    # Save model & print classification report
    model.save("snn_contrastive.keras")
    print("Model saved to snn_contrastive.keras")
    print(classification_report(y_test, pred_labels,
                                target_names=["Dissimilar(0)", "Similar(1)"]))

    # Additional metrics for logging
    metrics = {
        "accuracy": accuracy,
        "roc_auc": float(roc_auc),
        "best_thresh_dist": best_thresh_dist,
        "best_thresh_sim": float(best_thresh_sim),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "tpr": tp / (tp + fn),
        "tnr": tn / (tn + fp),
        "fpr": fp / (fp + tn),
        "fnr": fn / (fn + tp),
        "precision_dis": float(precision_0),
        "recall_dis": float(recall_0),
        "f1_dissimilar": float(f1_0),
        "precision_sim": float(precision_1),
        "recall_sim": float(recall_1),
        "f1_similar": float(f1_1),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "holdout_category": holdout_category,
        "holdout_accuracy": holdout_accuracy,
        "holdout_correct": int(total_correct),
        "holdout_total": int(total_samples),
        "test_eval_time(s)": float(test_eval_time),
    }

    return metrics
