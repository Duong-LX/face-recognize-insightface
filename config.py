# ========================
# Config cho Face Recognition Pipeline
# ========================

# Folder containing the original images collected
DATASET_DIR = "dataset"

# Folder containing image after aligning
ALIGNED_DIR = "aligned_dataset"

# Saved embeddings file
EMB_PATH = "faces_embeddings.npz"

# Model InsightFace
INSIGHTFACE_MODEL = "buffalo_l"

# Using GPU (CUDA)?
USE_GPU = False

# Where to save trained SVM model
MODEL_PATH = "svm_face_recognizer.joblib"

# Class list storage location (person name)
CLASSES_PATH = "svm_classes.npy"

# ----------------------
# Landmarks & Debug
# ----------------------
LANDMARKS_DIR = "landmarks"           # Directory to save JSON files of landmarks (kps)
DEBUG_DIR = "aligned_dataset_debug"   # Folder to save debug image (original image + landmarks)
SAVE_LANDMARKS = True                 # Save landmarks in preprocess step
SAVE_DEBUG = True                     # Save debug image in preprocess step

# ----------------------
# Training & Inference
# ----------------------
THRESHOLD = 0.7
