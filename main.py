import argparse
import numpy as np
from config import EMB_PATH
from register_user_camera import register_user_camera
from recognize import recognize_images, recognize_camera
from train_svm import train_svm
from validate_cosine_threshold import validate_threshold
from verify_onnx import verify_onnx


def main():
    parser = argparse.ArgumentParser(description="Face Recognition Pipeline CLI")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- Register ----
    reg_parser = subparsers.add_parser("register", help="Đăng ký user mới bằng camera/video")
    reg_parser.add_argument("--user", type=str, required=True, help="Tên user mới (vd: duong)")
    reg_parser.add_argument("--samples", type=int, default=50, help="Số ảnh cần thu thập")
    reg_parser.add_argument("--video", type=str, default="0", help="Nguồn video: 0 = webcam, hoặc đường dẫn video")

    # ---- Recognize ----
    rec_parser = subparsers.add_parser("recognize", help="Nhận diện khuôn mặt")
    rec_parser.add_argument("--images", nargs="+", help="Danh sách ảnh để nhận diện")
    rec_parser.add_argument("--camera", action="store_true", help="Dùng camera thay vì ảnh")

    # ---- Retrain SVM ----
    subparsers.add_parser("train", help="Train lại SVM từ embeddings")

    # ---- Validate threshold ----
    subparsers.add_parser("validate", help="Tính toán threshold cosine similarity tối ưu")

    # ---- Verify ONNX ----
    subparsers.add_parser("verify", help="So sánh Sklearn vs ONNX outputs")

    args = parser.parse_args()

    if args.command == "register":
        try:
            video_source = int(args.video)
        except ValueError:
            video_source = args.video
        register_user_camera(args.user, args.samples, video_source)

    elif args.command == "recognize":
        if args.camera:
            recognize_camera()
        elif args.images:
            recognize_images(args.images)
        else:
            print("[ERROR] Cần truyền --images hoặc --camera")

    elif args.command == "train":
        data = np.load(EMB_PATH)
        X, y = data["X"], data["y"]
        train_svm(X, y)

    elif args.command == "validate":
        validate_threshold()

    elif args.command == "verify":
        verify_onnx()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
