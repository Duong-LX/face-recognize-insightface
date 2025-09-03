import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from config import EMB_PATH, MODEL_PATH, CLASSES_PATH


def find_best_threshold(X_val, y_val, model):
    """Tìm threshold tối ưu bằng cách maximize F1-score trên validation."""
    probs = model.predict_proba(X_val)
    best_thr = 0.5
    best_f1 = 0

    for thr in np.linspace(0.1, 0.9, 81):  # sweep 0.1 -> 0.9
        y_pred = []
        for p in probs:
            conf = np.max(p)
            if conf < thr:
                y_pred.append(-1)  # -1 = others
            else:
                y_pred.append(np.argmax(p))

        f1 = f1_score(y_val, y_pred, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return best_thr, best_f1


def update_config_file(threshold):
    """Ghi threshold tối ưu vào config.py."""
    config_file = "config.py"
    lines = []
    with open(config_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("THRESHOLD"):
                lines.append(f"THRESHOLD = {threshold:.3f}\n")
            else:
                lines.append(line)
    with open(config_file, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"✅ Updated config.py → THRESHOLD = {threshold:.3f}")


def main():
    # Load embeddings
    data = np.load(EMB_PATH, allow_pickle=True)
    X, y = data["X"], data["y"]

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    print(f"🔄 Training SVM on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples")

    # Train SVM
    clf = SVC(kernel="linear", probability=True)
    clf.fit(X_train, y_train)

    # Find best threshold on validation
    best_thr, best_f1 = find_best_threshold(X_val, y_val, clf)
    print(f"🔎 Best threshold = {best_thr:.3f}, F1 = {best_f1:.4f}")

    # Save final model trained on ALL data
    print("🔄 Retraining final SVM on full dataset...")
    clf.fit(X, y_enc)
    joblib.dump(clf, MODEL_PATH)
    np.save(CLASSES_PATH, le.classes_)

    print(f"💾 Saved model → {MODEL_PATH}")
    print(f"💾 Saved classes → {CLASSES_PATH}")

    # Update config.py
    update_config_file(best_thr)


if __name__ == "__main__":
    main()
