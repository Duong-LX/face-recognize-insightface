import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from config import EMB_PATH, MODEL_PATH, CLASSES_PATH


def main():
    # Load embeddings
    data = np.load(EMB_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    print(f"✅ Loaded embeddings from {EMB_PATH}")
    print(f"   → {X.shape[0]} samples, {X.shape[1]}-dim features")

    # Encode labels
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    # Train SVM classifier
    print("🔄 Training SVM classifier (linear kernel, probability=True)...")
    clf = SVC(kernel="linear", probability=True)

    # Cross-validation
    scores = cross_val_score(clf, X, y_enc, cv=5)
    print(f"📊 Cross-validation accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # Fit final model on all data
    clf.fit(X, y_enc)

    # Save model
    joblib.dump(clf, MODEL_PATH)
    np.save(CLASSES_PATH, label_encoder.classes_)

    print(f"💾 Saved SVM model to {MODEL_PATH}")
    print(f"💾 Saved classes to {CLASSES_PATH}")
    print("🎉 Done!")


if __name__ == "__main__":
    main()
