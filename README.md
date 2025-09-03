
# Face Recognition with InsightFace + SVM (Full Pipeline)

Một project hoàn chỉnh cho nhận diện khuôn mặt sử dụng **InsightFace** (trích xuất embedding) và **SVM** (phân loại danh tính). Hỗ trợ:
- Thu thập & tiền xử lý dữ liệu (detect + align + embedding)
- Huấn luyện SVM
- Nhận diện từ ảnh, video, hoặc webcam
- Test face detection độc lập

## 1) Cài đặt

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> Ghi chú:
> - `CTX_ID` trong `config.py`: đặt `-1` để chạy CPU; `0` để dùng GPU (nếu đã setup).
> - Nếu dùng GPU với Windows, cân nhắc cài `onnxruntime-gpu` thay cho `onnxruntime`.

## 2) Cấu trúc dữ liệu

```
dataset/
├─ person_a/
│  ├─ 001.jpg
│  ├─ 002.jpg
├─ person_b/
   ├─ 001.jpg
```

## 3) Tiền xử lý & Tạo Embeddings

```bash
python preprocess_dataset.py
```
- Tạo thư mục `dataset_aligned/` (ảnh đã align).
- Tạo file `faces_embeddings.npz` (X: embeddings, y: labels).

## 4) Huấn luyện SVM

```bash
python train_svm.py
```
- Sinh `svm_face_recognizer.joblib` và `svm_classes.npy`.

## 5) Nhận diện

### 5.1 Ảnh
```bash
python recognize_image.py path/to/image.jpg
```

### 5.2 Video/Webcam
```bash
python recognize_video.py           # webcam (0)
python recognize_video.py 0         # chỉ định webcam id
python recognize_video.py path/to/video.mp4
```

Nhấn **Q** để thoát cửa sổ.

## 6) Test Face Detection (không cần SVM)
```bash
# Từ ảnh
python face_detector.py path/to/image.jpg

# Từ webcam (hoặc truyền số khác để chọn camera id)
python face_detector.py 0
```

## 7) Cấu hình (`config.py`)

```python
MODEL_NAME = 'buffalo_l'   # InsightFace model pack
CTX_ID = -1                # -1: CPU, 0: GPU
DET_SIZE = (640, 640)      # size đầu vào của detector
PROB_THRESHOLD = 0.55      # ngưỡng xác suất để gán 'Others'
MAX_FACES = 10             # giới hạn số mặt xử lý/mỗi frame
```

## 8) Khắc phục sự cố

- **TypeError: get_embedding_from_face() missing 1 required positional argument**  
  → Dự án đã chuẩn hoá hàm này trong `face_detector.py` với chữ ký `get_embedding_from_face(img, face_obj)`.
- **Không đọc được ảnh**: Kiểm tra đường dẫn & định dạng.  
- **Không phát hiện khuôn mặt**: Tăng sáng ảnh, đổi `DET_SIZE`, thử ảnh rõ hơn.  
- **Xác suất thấp, bị gán Others**: Tăng số ảnh/ người, hoặc hạ `PROB_THRESHOLD`.  
- **Hiệu năng chậm**: Điều chỉnh `MAX_FACES`, dùng GPU (`CTX_ID=0`).

## 9) Ghi chú triển khai
- SVM dùng pipeline `StandardScaler(with_mean=False) + SVC(kernel="linear", probability=True)`.
- Embedding được chuẩn hoá ngầm bởi InsightFace (tuỳ model). Bạn có thể thêm chuẩn hoá L2 nếu cần.
- Có thể thay `SVC` bằng `KNN/LogisticRegression` nếu dataset nhỏ.

---

**Pipeline chuẩn:**

1. Đặt ảnh vào `dataset/<person_name>/*.jpg`
2. `python preprocess_dataset.py`
3. `python train_svm.py`
4. `python recognize_image.py <img>` hoặc `python recognize_video.py`

Chúc bạn build hệ thống vui vẻ! 🚀

## 2) Thu thập dữ liệu (webcam/video)

```bash
# Thu thập 50 ảnh đã align từ webcam mặc định (0)
python data_collection.py --person Alice --source 0 --num 50 --aligned

# Thu thập từ file video, lưu crop bbox thay vì align
python data_collection.py --person Bob --source path/to/video.mp4 --num 80 --no-aligned

# Tuỳ chọn khác
# --out dataset2         # thay thư mục dataset gốc
# --every 3              # lưu 1 ảnh mỗi 3 frame khi có mặt
# --no-mirror            # không lật ngang webcam
```
Sau khi thu thập, ảnh sẽ nằm tại `dataset/<person>/*.jpg` (hoặc thư mục bạn chỉ định bằng `--out`).


(Thu thập ảnh) → dataset/<person>/*.jpg
        │
        ▼
[preprocess_dataset.py]
  - detect face (InsightFace buffalo_l)
  - align (112x112) + (optional) save landmarks/debug
  - extract embedding (face.normed_embedding)
  - save faces_embeddings.npz (X, y)
        │
        ▼
[train_svm.py]
  - train SVC(linear, prob=True)
  - save svm_face_recognizer.joblib + svm_classes.npy
        │
        ├── [test_threshold.py] → tìm THRESHOLD tối ưu → cập nhật config.py
        ▼
[recognize.py] (ảnh/video/camera)
  - detect → embedding → SVM prob
  - if prob < THRESHOLD → label = "others"
  - overlay kết quả (+ lưu *_recognized.jpg nếu là ảnh)



1) Tổng quan kiến trúc & artefacts chính

Thư mục & file quan trọng (mình đã đếm nhanh dữ liệu hiện có):

dataset/ → ảnh thô đã thu thập (hiện có 3 người, 31 ảnh: Dung, Duong, Huy)

aligned_dataset/ → ảnh đã align (cùng 3 người, 31 ảnh)

aligned_dataset_debug/ → debug ảnh gốc + landmarks (3 người, 31 ảnh)

landmarks/ → JSON landmarks (nếu bật)

faces_embeddings.npz → file embeddings (X: vectors, y: labels)

svm_face_recognizer.joblib → SVM đã train

svm_classes.npy → danh sách class labels

svm_face_recognizer.onnx → bản convert ONNX

Code chính:

config.py → cấu hình toàn pipeline (đường dẫn, model name, threshold, …)

face_detector.py → detector + align + test (ảnh/video/camera)

preprocess_dataset.py → detect → align → trích xuất embedding → lưu faces_embeddings.npz

train_svm.py → train SVM (linear, probability=True) + lưu model/labels

test_threshold.py → tối ưu THRESHOLD (F1-score trên validation) + cập nhật config.py

recognize.py → nhận diện ảnh/video/camera, gắn nhãn ‘others’ nếu < THRESHOLD, lưu ảnh kết quả

tests.py → convert SVM → ONNX và verify (so khớp dự đoán)

utils/draw.py → vẽ box/text/landmarks

Lưu ý nhỏ: data_collection.py có gọi utils.draw.draw_faces nhưng trong utils/draw.py hiện chưa có hàm này. Nếu bạn dùng data_collection.py, nên:

thêm hàm draw_faces hoặc

thay bằng draw_box + draw_landmarks như các file khác.

2) Cấu hình (config.py) — các key quan trọng

(đã trích xuất được các giá trị chính từ file)

DATASET_DIR = "dataset" – thư mục ảnh thô thu thập

ALIGNED_DIR = "aligned_dataset" – thư mục ảnh đã align

EMB_PATH = "faces_embeddings.npz" – nơi lưu embeddings

INSIGHTFACE_MODEL = "buffalo_l" – backbone InsightFace

USE_GPU = True – dùng GPU nếu khả dụng

MODEL_PATH = "svm_face_recognizer.joblib" – SVM đã train

CLASSES_PATH = "svm_classes.npy" – danh sách classes

LANDMARKS_DIR = "landmarks" – nơi lưu JSON keypoints

DEBUG_DIR = "aligned_dataset_debug" – nơi lưu ảnh debug

SAVE_LANDMARKS = True, SAVE_DEBUG = True – bật ghi file phụ trợ

THRESHOLD = 0.100 – ngưỡng nhận diện “others” (sẽ được tối ưu bằng test_threshold.py)

3) Pipeline chi tiết (end-to-end)
A. (Tuỳ chọn) Thu thập dữ liệu

Script: data_collection.py

Ý tưởng: mở camera/video/ảnh, detect và lưu samples đều đặn vào dataset/<person>/...

Flags chính (theo code):
--source (camera index / video path), --person (tên người), --out (thư mục đích, mặc định dataset), --num (số ảnh), --every (mỗi N frame lưu 1 ảnh), --aligned/--no-aligned, --no-mirror

Như lưu ý ở trên, cần sửa draw_faces nếu sử dụng ngay file này.

B. Preprocess → Align → Embedding

Script: preprocess_dataset.py

Logic:

Duyệt dataset/<person>/*.jpg|png|...

FaceDetector (InsightFace buffalo_l) → detect face tốt nhất trong ảnh

Align bằng insightface.utils.face_align (chuẩn 112×112)

(Tuỳ chọn) Lưu landmarks (JSON) & ảnh debug (bật bởi SAVE_*)

Embedding: lấy face.normed_embedding (đã L2-normalized)

Thu thập X (embeddings), y (person) → lưu faces_embeddings.npz

Chạy: