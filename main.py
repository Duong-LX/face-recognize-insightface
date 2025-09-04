import argparse
import numpy as np
from config import EMB_PATH
from register_user_camera import register_user_camera
from recognize import run_images, run_video, FaceRecognizer
from train_svm import main as train_svm_main
import verify_onnx


def main():
    parser = argparse.ArgumentParser(description="Face Recognition Pipeline CLI")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- Register ----
    reg_parser = subparsers.add_parser("register", help="Register user mới bằng camera/video")
    reg_parser.add_argument("--user", type=str, required=True, help="New username (vd: duong)")
    reg_parser.add_argument("--samples", type=int, default=50, help="Number images to collect")
    reg_parser.add_argument("--video", type=str, default="0", help="Video source: 0 = webcam, or video path")
    reg_parser.add_argument("--no_save_video", action="store_true", help="Don't save output video")

    # ---- Recognize ----
    rec_parser = subparsers.add_parser("recognize", help="Face Recognize ")
    rec_parser.add_argument("--images", nargs="+", help="List of images to recognize")
    rec_parser.add_argument("--video", type=str, help="Video path")
    rec_parser.add_argument("--camera", type=int, help="camera index (vd: 0)")
    rec_parser.add_argument("--no_save", action="store_true", help="Don't save output video")

    # ---- Retrain SVM ----
    subparsers.add_parser("train", help="Retraining SVM from embeddings")

    # ---- Verify ONNX ----
    subparsers.add_parser("verify", help="Compare Sklearn vs ONNX outputs")

    args = parser.parse_args()

    if args.command == "register":
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

    elif args.command == "recognize":
        recognizer = FaceRecognizer()
        if args.images:
            run_images(recognizer, args.images, save_output=not args.no_save)
        elif args.video:
            run_video(recognizer, args.video)
        elif args.camera is not None:
            run_video(recognizer, args.camera)
        else:
            print("[ERROR] You must provide --images, --video or --camera")

    elif args.command == "train":
        data = np.load(EMB_PATH)
        X, y = data["X"], data["y"]
        print(f"[INFO] Retraining SVM with {len(X)} samples...")
        train_svm_main()

    elif args.command == "verify":
        verify_onnx()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
