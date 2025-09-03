import numpy as np
import joblib
from config import EMB_PATH, MODEL_PATH, CLASSES_PATH
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import os


def convert_to_onnx():
    # Load sklearn SVM
    clf = joblib.load(MODEL_PATH)
    n_features = clf.support_vectors_.shape[1]

    # Convert to ONNX
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type)

    onnx_path = os.path.splitext(MODEL_PATH)[0] + ".onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"✅ Saved ONNX model to {onnx_path}")

    return onnx_path


def verify(onnx_path):
    # Load sklearn model
    clf = joblib.load(MODEL_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)

    # Load embeddings (demo test trên vài samples)
    data = np.load(EMB_PATH, allow_pickle=True)
    X, y = data["X"], data["y"]

    # Pick 5 samples
    X_test = X[:5]

    # Sklearn predictions
    probs_sklearn = clf.predict_proba(X_test)

    # ONNX predictions
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[1].name  # [0] = label, [1] = probability
    probs_onnx = sess.run([output_name], {input_name: X_test.astype(np.float32)})[0]

    # ✅ Convert dict → numpy
    if isinstance(probs_onnx[0], dict):
        probs_onnx = np.array([list(d.values()) for d in probs_onnx])
    else:
        probs_onnx = np.array(probs_onnx)

    # Compare
    print("\n🔎 Verify predictions (Sklearn vs ONNX):")
    for i in range(len(X_test)):
        print(f"Sample {i}:")
        print(f"  Sklearn → {classes[np.argmax(probs_sklearn[i])]} "
              f"({np.max(probs_sklearn[i]):.4f})")
        print(f"  ONNX    → {classes[np.argmax(probs_onnx[i])]} "
              f"({np.max(probs_onnx[i]):.4f})")



if __name__ == "__main__":
    onnx_path = convert_to_onnx()
    verify(onnx_path)
