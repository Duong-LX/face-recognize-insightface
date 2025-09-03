import os
import cv2
import numpy as np
import json
from face_detector import FaceDetector
from config import DATASET_DIR, ALIGNED_DIR, EMB_PATH, LANDMARKS_DIR, DEBUG_DIR, SAVE_LANDMARKS, SAVE_DEBUG
from utils.draw import draw_landmarks
from insightface.utils import face_align

os.makedirs(ALIGNED_DIR, exist_ok=True)
if SAVE_LANDMARKS:
    os.makedirs(LANDMARKS_DIR, exist_ok=True)
if SAVE_DEBUG:
    os.makedirs(DEBUG_DIR, exist_ok=True)

def preprocess_dataset():
    detector = FaceDetector()
    invalid = []
    X, y = [], []

    for person in os.listdir(DATASET_DIR):
        p_in = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(p_in):
            continue

        # chuẩn bị output dirs
        p_out_aligned = os.path.join(ALIGNED_DIR, person)
        os.makedirs(p_out_aligned, exist_ok=True)
        p_out_landmarks = os.path.join(LANDMARKS_DIR, person) if SAVE_LANDMARKS else None
        if SAVE_LANDMARKS:
            os.makedirs(p_out_landmarks, exist_ok=True)
        p_out_debug = os.path.join(DEBUG_DIR, person) if SAVE_DEBUG else None
        if SAVE_DEBUG:
            os.makedirs(p_out_debug, exist_ok=True)

        for fname in os.listdir(p_in):
            fpath = os.path.join(p_in, fname)
            if not os.path.isfile(fpath):
                continue

            try:
                img = cv2.imread(fpath)
                if img is None:
                    invalid.append((fpath, "cv2.imread failed"))
                    continue

                faces = detector.detect_faces(img)
                if not faces:
                    invalid.append((fpath, "no face"))
                    continue

                face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
                kps = face.kps  # (5, 2)
                aligned = face_align.norm_crop(img, kps)

                # save aligned
                out_aligned_path = os.path.join(p_out_aligned, fname)
                cv2.imwrite(out_aligned_path, aligned)

                # save landmarks
                if SAVE_LANDMARKS:
                    lm_path = os.path.join(p_out_landmarks, os.path.splitext(fname)[0] + ".json")
                    with open(lm_path, "w", encoding="utf-8") as f:
                        json.dump(kps.astype(float).tolist(), f)

                # save debug image
                if SAVE_DEBUG:
                    debug_img = img.copy()
                    debug_img = draw_landmarks(debug_img, kps)
                    out_debug_path = os.path.join(p_out_debug, fname)
                    cv2.imwrite(out_debug_path, debug_img)

                # embedding
                emb = face.normed_embedding
                if emb is None:
                    invalid.append((fpath, "no embedding"))
                    continue

                X.append(emb)
                y.append(person)

            except Exception as e:
                invalid.append((fpath, f"exception: {e}"))

    X = np.array(X)
    y = np.array(y)
    np.savez(EMB_PATH, X=X, y=y)

    print(f"✅ Preprocess xong! Saved {EMB_PATH} với {len(X)} samples, {len(set(y))} classes.")
    if invalid:
        print("⚠️ Các file lỗi:")
        for item in invalid:
            print(" -", item[0], "=>", item[1])

if __name__ == "__main__":
    preprocess_dataset()
