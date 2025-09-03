# ========================
# Config cho Face Recognition Pipeline
# ========================

# Thư mục chứa ảnh gốc thu thập
DATASET_DIR = "dataset"

# Thư mục chứa ảnh sau khi align
ALIGNED_DIR = "aligned_dataset"

# File embeddings đã lưu
EMB_PATH = "faces_embeddings.npz"

# Model InsightFace
INSIGHTFACE_MODEL = "buffalo_l"

# Sử dụng GPU (CUDA) hay không
USE_GPU = True

# Nơi lưu model SVM đã train
MODEL_PATH = "svm_face_recognizer.joblib"

# Nơi lưu danh sách lớp (tên người)
CLASSES_PATH = "svm_classes.npy"

# ----------------------
# Landmarks & Debug
# ----------------------
LANDMARKS_DIR = "landmarks"           # Thư mục lưu file JSON các điểm mốc (kps)
DEBUG_DIR = "aligned_dataset_debug"   # Thư mục lưu ảnh debug (ảnh gốc + landmarks)
SAVE_LANDMARKS = True                 # Lưu landmarks trong bước preprocess
SAVE_DEBUG = True                     # Lưu ảnh debug trong bước preprocess

# ----------------------
# Training & Inference
# ----------------------
THRESHOLD = 0.100
