# Face Recognition (InsightFace + SVM)

Pipeline nhận diện khuôn mặt sử dụng **InsightFace (ArcFace embeddings)** kết hợp với **SVM classifier**.  
Hỗ trợ đăng ký user mới, nhận diện real-time, tinh chỉnh threshold và export sang ONNX.

---

## 🚀 Cài đặt

```bash
git clone <repo-url>
cd face_recognition_full_insightface_project_complete_update
pip install -r requirements.txt
```

---

## 📂 Cấu trúc chính

```
.
├── config.py                   # Config chung (đường dẫn, threshold, model path)
├── main.py                     # CLI quản lý toàn bộ pipeline
├── register_user_camera.py     # Đăng ký user mới qua camera/video
├── recognize.py                # Nhận diện từ ảnh hoặc camera
├── preprocess_dataset.py       # Preprocess + align + embedding dataset
├── train_svm.py                # Train SVM
├── validate_cosine_threshold.py# Tối ưu threshold cosine similarity
├── verify_onnx.py              # Chuyển & verify ONNX
├── utils/
│   └── draw.py                 # Hàm hỗ trợ vẽ bounding box, label
└── requirements.txt
```

---

## ⚡ Sử dụng qua `main.py`

### 1. Đăng ký user mới
Thu thập ảnh từ camera hoặc video, preprocess và train lại SVM:

```bash
# Từ webcam
python main.py register --user duong --samples 60 --video 0

# Từ video
python main.py register --user duong --samples 60 --video duong.mp4
```

---

### 2. Nhận diện khuôn mặt

- Nhận diện từ ảnh:
```bash
python main.py recognize --images samples/a.jpg samples/b.jpg
```

- Nhận diện từ camera:
```bash
python main.py recognize --camera
```

---

### 3. Train lại SVM (toàn bộ dataset)

```bash
python main.py train
```

---

### 4. Tối ưu threshold (cosine similarity)

```bash
python main.py validate
```

---

### 5. Verify ONNX (so sánh Sklearn vs ONNX)

```bash
python main.py verify
```

---

## 📌 Ghi chú
- Dataset lưu tại `dataset/<username>/`  
- Embeddings: `faces_embeddings.npz`  
- Model SVM: `svm_face_recognizer.joblib`, `svm_classes.npy`  
- ONNX model: `svm_face_recognizer.onnx`  
- Threshold được lưu trong `config.py`

---

## 🛡️ Yêu cầu
- Python 3.8+
- OpenCV
- InsightFace
- Scikit-learn
- ONNX, onnxruntime
- joblib
