import cv2
import os
import argparse
import numpy as np
from config import DATASET_DIR, EMB_PATH, MODEL_PATH, CLASSES_PATH
from preprocess_dataset import preprocess_dataset
from train_svm import main as train_svm_main
from face_detector import FaceDetector


def register_user_camera(username, num_samples=50, video_source=0, save_video=True):
    """
    Register new user with camera or video.
    :param username: New user name
    :param num_samples: Number of images to collect
    :param video_source: 0 = webcam, video path
    :param save_video: Save video output
    """
    user_path = os.path.join(DATASET_DIR, username)
    os.makedirs(user_path, exist_ok=True)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video/camera: {video_source}")
        return

    # Writer video output
    out = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = f"{username}_registration.mp4"
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"[INFO] Output video saved in: {out_path}")

    detector = FaceDetector()
    count = 0

    print(f"[INFO] Start collecting data for user '{username}'...")

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)
        for face in faces:
            aligned = detector.align_face(frame, face, size=112)
            face_path = os.path.join(user_path, f"{username}_{count}.jpg")
            cv2.imwrite(face_path, aligned)
            count += 1
            print(f"[INFO] Saved {count}/{num_samples} image for {username}")
            if count >= num_samples:
                break

        # Show frame
        cv2.imshow("Register User", frame)
        if save_video and out:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    if count == 0:
        print("[ERROR] Cannot collect any faces!")
        return

    # Preprocess all dataset (include new user)
    print("[INFO] Preprocessing dataset...")
    preprocess_dataset()

    # Retraining SVM
    print("[INFO] Retraining SVM with new dataset...")
    train_svm_main()

    print(f"[SUCCESS] Register user '{username}' completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register user with camera or video")
    parser.add_argument("--user", type=str, required=True, help="New username (ex: Duong)")
    parser.add_argument("--samples", type=int, default=50, help="Number images to collect")
    parser.add_argument("--video", type=str, default="0", help="Video source: 0 = webcam, or video path")
    parser.add_argument("--no_save_video", action="store_true", help="Don't save output video")

    args = parser.parse_args()

    # If --video is number, convert to int for cv2.VideoCapture
    try:
        video_source = int(args.video)
    except ValueError:
        video_source = args.video

    register_user_camera(
        args.user,
        num_samples=args.samples,
        video_source=video_source,
        save_video=not args.no_save_video,
    )
