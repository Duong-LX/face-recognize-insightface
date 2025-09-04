import cv2
import os
import argparse
import numpy as np
from config import DATASET_PATH, EMB_PATH
from preprocess_dataset import preprocess_images
from train_svm import train_svm
from face_detector import detect_and_align_faces


def register_user_camera(username, num_samples=50, video_source=0):
    """
    Đăng ký user mới qua camera hoặc video
    :param username: tên user mới
    :param num_samples: số lượng ảnh khuôn mặt cần thu thập
    :param video_source: 0 = webcam, hoặc đường dẫn video
    """

    user_path = os.path.join(DATASET_PATH, username)
    os.makedirs(user_path, exist_ok=True)

    cap = cv2.VideoCapture(video_source)
    count = 0

    print(f"[INFO] Bắt đầu thu thập dữ liệu cho user '{username}'...")

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_and_align_faces(frame)
        for face in faces:
            face_path = os.path.join(user_path, f"{username}_{count}.jpg")
            cv2.imwrite(face_path, face)
            count += 1
            print(f"[INFO] Đã lưu {count}/{num_samples} ảnh cho {username}")
            if count >= num_samples:
                break

        # Hiển thị frame
        cv2.imshow("Register User", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count == 0:
        print("[ERROR] Không thu thập được ảnh khuôn mặt nào!")
        return

    # Preprocess dữ liệu user mới
    print("[INFO] Đang preprocess ảnh...")
    preprocess_images([username])  # chỉ xử lý user mới

    # Load toàn bộ embeddings
    data = np.load(EMB_PATH)
    X, y = data["X"], data["y"]

    # Train lại SVM
    print("[INFO] Train lại SVM...")
    train_svm(X, y)

    print(f"[SUCCESS] Đăng ký user '{username}' hoàn tất!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đăng ký user mới qua camera hoặc video")
    parser.add_argument("--user", type=str, required=True, help="Tên user mới (ví dụ: duong)")
    parser.add_argument("--samples", type=int, default=50, help="Số lượng ảnh khuôn mặt cần thu thập")
    parser.add_argument("--video", type=str, default="0", help="Nguồn video: 0 = webcam, hoặc đường dẫn file video")

    args = parser.parse_args()

    # Nếu --video là số, chuyển về int để dùng cho cv2.VideoCapture
    try:
        video_source = int(args.video)
    except ValueError:
        video_source = args.video

    register_user_camera(args.user, num_samples=args.samples, video_source=video_source)
