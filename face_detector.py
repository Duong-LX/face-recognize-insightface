import argparse
import cv2
import os
import time
import insightface
import numpy as np
from config import INSIGHTFACE_MODEL, USE_GPU
from utils.draw import draw_box, draw_text, draw_landmarks
from insightface.utils import face_align


class FaceDetector:
    def __init__(self, model_name=None, det_size=480):
        model_name = model_name or INSIGHTFACE_MODEL
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if USE_GPU else ["CPUExecutionProvider"]
        self.model = insightface.app.FaceAnalysis(name=model_name, providers=providers)
        self.model.prepare(ctx_id=0 if USE_GPU else -1, det_size=(det_size, det_size))

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
        return face.kps

    def align_face(self, img, face, size=112):
        return face_align.norm_crop(img, face.kps, image_size=size)


# ------------------------
# CLI test tool + benchmark
# ------------------------
def run_images(detector, image_paths, save_aligned=False):
    os.makedirs("outputs/images", exist_ok=True)
    total_faces, total_imgs = 0, 0
    start_time = time.time()

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"❌ Cannot read image {path}")
            continue
        total_imgs += 1
        faces = detector.detect_faces(img)
        total_faces += len(faces)

        for i, face in enumerate(faces):
            bbox = face.bbox
            draw_box(img, bbox)
            draw_landmarks(img, face.kps)
            draw_text(img, f"{face.det_score:.2f}", (int(bbox[0]), int(bbox[1]) - 10))

            if save_aligned:
                aligned = detector.align_face(img, face, size=112)
                out_path = f"outputs/images/{os.path.splitext(os.path.basename(path))[0]}_aligned_{i}.jpg"
                cv2.imwrite(out_path, aligned)
                print(f"💾 Saved aligned face: {out_path}")

        out_path = f"outputs/images/{os.path.splitext(os.path.basename(path))[0]}_detected.jpg"
        cv2.imwrite(out_path, img)
        print(f"💾 Saved result image: {out_path}")

        cv2.imshow("Result", img)
        cv2.waitKey(0)

    elapsed = time.time() - start_time
    cv2.destroyAllWindows()

    print("=== BENCHMARK: DETECTION (IMAGES) ===")
    print(f"📂 Total images: {total_imgs}")
    print(f"🙂 Total faces detected: {total_faces}")
    print(f"⏱️ Time: {elapsed:.2f}s")
    print(f"⚡ Speed: {total_faces / elapsed:.2f} faces/sec" if elapsed > 0 else "⚡ Speed: 0")
    print("=====================================")


def run_video(detector, source=0, save_aligned=False):
    os.makedirs("outputs/videos", exist_ok=True)
    cap = cv2.VideoCapture(source)
    frame_idx, total_faces = 0, 0
    start_time = time.time()

    # Output path
    if isinstance(source, str):
        video_name = os.path.splitext(os.path.basename(source))[0]
    else:
        video_name = f"camera{source}"
    out_path = f"outputs/videos/{video_name}_detected.mp4"

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detector.detect_faces(frame)
        total_faces += len(faces)
        for i, face in enumerate(faces):
            bbox = face.bbox
            draw_box(frame, bbox)
            draw_landmarks(frame, face.kps)
            draw_text(frame, f"{face.det_score:.2f}", (int(bbox[0]), int(bbox[1]) - 10))

            if save_aligned:
                aligned = detector.align_face(frame, face, size=112)
                aligned_path = f"outputs/images/{video_name}_frame{frame_idx}_face{i}.jpg"
                cv2.imwrite(aligned_path, aligned)

        out_writer.write(frame)
        cv2.imshow("Result", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
        frame_idx += 1

    elapsed = time.time() - start_time
    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()

    fps = frame_idx / elapsed if elapsed > 0 else 0
    print("=== BENCHMARK: DETECTION (VIDEO) ===")
    print(f"🎞️ Frames processed: {frame_idx}")
    print(f"🙂 Total faces detected: {total_faces}")
    print(f"⏱️ Time: {elapsed:.2f}s")
    print(f"⚡ FPS: {fps:.2f}")
    print(f"💾 Saved output video: {out_path}")
    print("====================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", help="Test on images")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--camera", type=int, help="Camera index, e.g., 0")
    parser.add_argument("--save_aligned", action="store_true", help="Save aligned faces while testing")
    args = parser.parse_args()

    detector = FaceDetector(det_size=480)

    if args.images:
        run_images(detector, args.images, save_aligned=args.save_aligned)
    elif args.video:
        run_video(detector, args.video, save_aligned=args.save_aligned)
    elif args.camera is not None:
        run_video(detector, args.camera, save_aligned=args.save_aligned)
    else:
        print("⚠️ Please provide --images, --video, or --camera")
