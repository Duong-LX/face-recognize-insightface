# validate_cosine_threshold.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import config

def compute_class_prototypes(X, y):
    """Compute mean embedding (prototype) per class."""
    prototypes = {}
    for label in np.unique(y):
        prototypes[label] = np.mean(X[y == label], axis=0)
    return prototypes

def validate_threshold(X_val, y_val, prototypes, thresholds):
    """Sweep thresholds and compute F1-score."""
    best_thr, best_f1 = 0, -1
    for thr in thresholds:
        y_pred = []
        for i, x in enumerate(X_val):
            sims = {label: cosine_similarity([x], [proto])[0][0]
                    for label, proto in prototypes.items()}
            best_label, best_sim = max(sims.items(), key=lambda kv: kv[1])
            if best_sim < thr:
                y_pred.append("others")
            else:
                y_pred.append(best_label)
        f1 = (np.mean(np.array(y_pred) == np.array(y_val))).item()  # simple accuracy
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

def main():
    # Load embeddings
    data = np.load(config.EMB_PATH, allow_pickle=True)
    X, y = data["X"], data["y"]

    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Compute prototypes on train
    prototypes = compute_class_prototypes(X_train, y_train)

    # Search thresholds
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thr, best_score = validate_threshold(X_val, y_val, prototypes, thresholds)
    print(f"[Cosine Validation] Best threshold = {best_thr:.3f}, Score = {best_score:.4f}")

    # Update config.py
    lines = []
    with open("config.py", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("THRESHOLD"):
                lines.append(f'THRESHOLD = {best_thr:.3f}\n')
            else:
                lines.append(line)
    with open("config.py", "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("✅ Updated config.py with best THRESHOLD")

    # Retrain SVM on full dataset (optional)
    clf = SVC(kernel="linear", probability=True)
    clf.fit(X, y)
    joblib.dump(clf, config.MODEL_PATH)
    np.save(config.CLASSES_PATH, np.unique(y))
    print(f"✅ Saved updated SVM model to {config.MODEL_PATH}")

if __name__ == "__main__":
    main()
