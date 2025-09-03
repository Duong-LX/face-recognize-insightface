import numpy as np
import joblib
import onnxruntime as ort
import time

from config import EMB_PATH, MODEL_PATH, CLASSES_PATH


def verify(onnx_path, n_samples=5):
    # Load Sklearn SVM + classes
    svm_model = joblib.load(MODEL_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)

    # Load embeddings
    data = np.load(EMB_PATH)
    X, y = data["X"], data["y"]

    # Predict with Sklearn
    probs_sklearn = svm_model.predict_proba(X[:n_samples])
    preds_sklearn = np.argmax(probs_sklearn, axis=1)

    # Load ONNX runtime
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[1].name  # usually "probabilities"

    # Predict with ONNX
    probs_onnx = session.run([output_name], {input_name: X[:n_samples].astype(np.float32)})[0]
    preds_onnx = np.argmax(probs_onnx, axis=1)

    # Compare results
    print("\n🔎 Verification Results (Sklearn vs ONNX):")
    for i in range(n_samples):
        print(f"Sample {i}:")
        print(f"  Sklearn → {classes[preds_sklearn[i]]} ({np.max(probs_sklearn[i]):.4f})")
        print(f"  ONNX    → {classes[preds_onnx[i]]} ({np.max(probs_onnx[i]):.4f})")
        print("-" * 40)

    # --- Benchmark Accuracy ---
    print("\n⚡ Benchmarking full dataset...")
    # Sklearn
    preds_sklearn_full = svm_model.predict(X)
    acc_sklearn = np.mean(preds_sklearn_full == y)

    # ONNX
    batch_size = 128
    preds_onnx_full = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i+batch_size].astype(np.float32)
        probs = session.run([output_name], {input_name: xb})[0]
        preds_onnx_full.extend(np.argmax(probs, axis=1))
    preds_onnx_full = np.array(preds_onnx_full)
    acc_onnx = np.mean(preds_onnx_full == y)

    print(f"Accuracy (Sklearn) = {acc_sklearn*100:.2f}%")
    print(f"Accuracy (ONNX)    = {acc_onnx*100:.2f}%")

    # --- Benchmark Speed ---
    n_runs = 1000
    X_test = X[:n_runs]

    # Sklearn timing
    start = time.time()
    svm_model.predict_proba(X_test)
    t_sklearn = (time.time() - start) / n_runs * 1000  # ms/sample

    # ONNX timing
    start = time.time()
    session.run([output_name], {input_name: X_test.astype(np.float32)})
    t_onnx = (time.time() - start) / n_runs * 1000

    print(f"\n⏱ Avg inference time / sample:")
    print(f"Sklearn = {t_sklearn:.4f} ms")
    print(f"ONNX    = {t_onnx:.4f} ms")


if __name__ == "__main__":
    onnx_model_path = "svm_face_recognizer.onnx"
    verify(onnx_model_path, n_samples=5)
