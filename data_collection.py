import os
import cv2
import argparse
import time
import config
from face_detector import FaceDetector
from utils import draw

def collect_from_camera(person, output_dir, num_samples=100, interval=5):
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    count = 0
    saved = 0

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Collecting images for '{person}'... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)

        if faces:
            labels = [person] * len(faces)
            vis = draw.draw_faces(frame.copy(), faces, labels)

            if count % interval == 0 and saved < num_samples:
                face = faces[0]  # chỉ lấy 1 mặt chính
                x1, y1, x2, y2 = map(int, face.bbox)
                face_crop = frame[y1:y2, x1:x2]
                out_path = os.path.join(output_dir, f"{saved:04d}.jpg")
                cv2.imwrite(out_path, face_crop)
                saved += 1
                print(f"[SAVE] {out_path}")

            cv2.imshow("Collecting", vis)
        else:
            cv2.imshow("Collecting", frame)

        count += 1
        if cv2.waitKey(1) & 0xFF == ord("q") or saved >= num_samples:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Collected {saved} samples for {person}.")


def collect_from_video(video_path, person, output_dir, num_samples=100, interval=5):
    cap = cv2.VideoCapture(video_path)
    detector = FaceDetector()
    count, saved = 0, 0

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Collecting images for '{person}' from video {video_path}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)

        if faces:
            labels = [person] * len(faces)
            vis = draw.draw_faces(frame.copy(), faces, labels)

            if count % interval == 0 and saved < num_samples:
                face = faces[0]
                x1, y1, x2, y2 = map(int, face.bbox)
                face_crop = frame[y1:y2, x1:x2]
                out_path = os.path.join(output_dir, f"{saved:04d}.jpg")
                cv2.imwrite(out_path, face_crop)
                saved += 1
                print(f"[SAVE] {out_path}")

            cv2.imshow("Collecting", vis)
        else:
            cv2.imshow("Collecting", frame)

        count += 1
        if cv2.waitKey(1) & 0xFF == ord("q") or saved >= num_samples:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Collected {saved} samples for {person} from video.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", type=str, required=True, help="Tên người cần thu thập")
    parser.add_argument("--video", type=str, default=None, help="Video file path (nếu có)")
    parser.add_argument("--num_samples", type=int, default=100, help="Số image muốn thu thập")
    parser.add_argument("--interval", type=int, default=5, help="Chu kỳ lưu (frame)")
    args = parser.parse_args()

    save_dir = os.path.join(config.DATASET_DIR, args.person)

    if args.video:
        collect_from_video(args.video, args.person, save_dir,
                           num_samples=args.num_samples, interval=args.interval)
    else:
        collect_from_camera(args.person, save_dir,
                            num_samples=args.num_samples, interval=args.interval)
