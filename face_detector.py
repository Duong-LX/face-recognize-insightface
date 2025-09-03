import argparse
import cv2
import os
import insightface
import numpy as np
from config import INSIGHTFACE_MODEL, USE_GPU
from utils.draw import draw_box, draw_text, draw_landmarks
from insightface.utils import face_align

class FaceDetector:
    def __init__(self, model_name=None):
        model_name = model_name or INSIGHTFACE_MODEL
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if USE_GPU else ["CPUExecutionProvider"]
        self.model = insightface.app.FaceAnalysis(name=model_name, providers=providers)
        self.model.prepare(ctx_id=0 if USE_GPU else -1)

    def detect_faces(self, img):
        return self.model.get(img)

    def get_embedding(self, img):
        faces = self.detect_faces(img)
        if not faces:
            return None
        face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
        return face.normed_embedding

    def get_embedding_from_face(self, img, face):
        return face.normed_embedding

    def get_landmarks(self, face):
        """Trả về landmarks (5 điểm) dạng np.ndarray shape (5, 2)."""
        return face.kps

    def align_face(self, img, face, size=112):
        """Align khuôn mặt theo 5 điểm landmarks, trả về ảnh crop chuẩn."""
        return face_align.norm_crop(img, face.kps, image_size=size)


# ------------------------
# CLI test tool
# ------------------------
def run_images(detector, image_paths, save_aligned=False):
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"❌ Cannot read image {path}")
            continue
        faces = detector.detect_faces(img)
        for i, face in enumerate(faces):
            bbox = face.bbox
            draw_box(img, bbox)
            kps = face.kps
            draw_landmarks(img, kps)
            draw_text(img, f"{face.det_score:.2f}", (int(bbox[0]), int(bbox[1])-10))

            if save_aligned:
                aligned = detector.align_face(img, face, size=112)
                out_path = f"{os.path.splitext(path)[0]}_aligned_{i}.jpg"
                cv2.imwrite(out_path, aligned)
                print(f"💾 Saved aligned face: {out_path}")

        cv2.imshow("Result", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_video(detector, source=0, save_aligned=False):
    cap = cv2.VideoCapture(source)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detector.detect_faces(frame)
        for i, face in enumerate(faces):
            bbox = face.bbox
            draw_box(frame, bbox)
            draw_landmarks(frame, face.kps)
            draw_text(frame, f"{face.det_score:.2f}", (int(bbox[0]), int(bbox[1])-10))

            if save_aligned:
                aligned = detector.align_face(frame, face, size=112)
                out_path = f"aligned_frame{frame_idx}_face{i}.jpg"
                cv2.imwrite(out_path, aligned)

        cv2.imshow("Result", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
        frame_idx += 1
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", help="Test on images")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--camera", type=int, help="Camera index, e.g., 0")
    parser.add_argument("--save_aligned", action="store_true", help="Save aligned faces while testing")
    args = parser.parse_args()

    detector = FaceDetector()

    if args.images:
        run_images(detector, args.images, save_aligned=args.save_aligned)
    elif args.video:
        run_video(detector, args.video, save_aligned=args.save_aligned)
    elif args.camera is not None:
        run_video(detector, args.camera, save_aligned=args.save_aligned)
    else:
        print("⚠️ Please provide --images, --video, or --camera")
