import argparse
import cv2
import joblib
import numpy as np
import os
from config import MODEL_PATH, CLASSES_PATH, THRESHOLD
from face_detector import FaceDetector
from utils.draw import draw_box, draw_text, draw_landmarks


class FaceRecognizer:
    def __init__(self):
        print("🔄 Loading SVM model and classes...")
        self.model = joblib.load(MODEL_PATH)
        self.classes = np.load(CLASSES_PATH, allow_pickle=True)
        self.detector = FaceDetector()
        self.threshold = THRESHOLD
        print(f"⚙️ Recognition threshold = {self.threshold:.3f}")

    def predict(self, img):
        faces = self.detector.detect_faces(img)
        results = []
        for face in faces:
            emb = self.detector.get_embedding_from_face(img, face)
            if emb is None:
                continue
            probs = self.model.predict_proba([emb])[0]
            idx = np.argmax(probs)
            confidence = probs[idx]

            if confidence < self.threshold:
                label = "others"
            else:
                label = self.classes[idx]

            results.append((face, label, confidence))
        return results


# ------------------------
# CLI test tool
# ------------------------
def run_images(recognizer, image_paths, save_output=True):
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"❌ Cannot read image {path}")
            continue
        results = recognizer.predict(img)
        for face, label, conf in results:
            bbox = face.bbox
            draw_box(img, bbox)
            draw_landmarks(img, face.kps)
            draw_text(img, f"{label} ({conf:.2f})", (int(bbox[0]), int(bbox[1]) - 10))

        cv2.imshow("Recognition", img)
        cv2.waitKey(0)

        if save_output:
            out_path = os.path.splitext(path)[0] + "_recognized.jpg"
            cv2.imwrite(out_path, img)
            print(f"💾 Saved output: {out_path}")

    cv2.destroyAllWindows()


def run_video(recognizer, source=0):
    cap = cv2.VideoCapture(source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = recognizer.predict(frame)
        for face, label, conf in results:
            bbox = face.bbox
            draw_box(frame, bbox)
            draw_landmarks(frame, face.kps)
            draw_text(frame, f"{label} ({conf:.2f})", (int(bbox[0]), int(bbox[1]) - 10))
        cv2.imshow("Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", help="Test recognition on images")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--camera", type=int, help="Camera index, e.g., 0")
    parser.add_argument("--no_save", action="store_true", help="Do not save output images")
    args = parser.parse_args()

    recognizer = FaceRecognizer()

    if args.images:
        run_images(recognizer, args.images, save_output=not args.no_save)
    elif args.video:
        run_video(recognizer, args.video)
    elif args.camera is not None:
        run_video(recognizer, args.camera)
    else:
        print("⚠️ Please provide --images, --video, or --camera")
