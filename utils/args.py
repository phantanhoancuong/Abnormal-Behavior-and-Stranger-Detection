import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time Face Detection and Recognition"
    )

    # Runtime settings
    parser.add_argument(
        "--mode",
        "-m",
        choices=["amp", "no_amp"],
        default="no_amp",
        help="AMP: Automatic Mixed Precision ('amp') or standard mode ('no_amp'). Default is 'no_amp'.",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default="0",
        help="Camera index (e.g., 0) or video file path. Default is '0' (webcam).",
    )

    # Model paths
    parser.add_argument(
        "--face-detector",
        type=str,
        default="assets/face_detectors/YOLOv11/yolov11s_face.pt",
        help="Path to YOLO face detector model weights.",
    )
    parser.add_argument(
        "--tracker-config",
        type=str,
        default="assets/configs/bytetrack.yaml",
        help="Path to tracker config file (default: ByteTrack YAML).",
    )
    parser.add_argument(
        "--face-bank-path",
        type=str,
        default="data/face_banks/face_bank.pkl",
        help="Path to face bank pickle file (default: 'data/face_banks/face_bank.pkl').",
    )
    parser.add_argument(
        "--face-recognizer",
        type=str,
        choices=[
            "edgeface_base",
            "edgeface_s_gamma_05",
            "edgeface_xs_gamma_06",
            "edgeface_xxs",
        ],
        default="edgeface_s_gamma_05",
        help="Which EdgeFace model to use for recognition (default: edgeface_s_gamma_05).",
    )

    # Detection and recognition thresholds
    parser.add_argument(
        "--det-threshold",
        type=float,
        default=0.5,
        help="Face detection confidence threshold (default: 0.5).",
    )
    parser.add_argument(
        "--rec-threshold",
        type=float,
        default=0.5,
        help="Face recognition similarity threshold (default: 0.5).",
    )
    parser.add_argument(
        "--blurry-threshold",
        type=float,
        default=60.0,
        help="Laplacian variance threshold for blur detection (default: 60.0). Lower values allow blurrier faces.",
    )

    # Tracker settings
    parser.add_argument(
        "--min-track-age",
        type=int,
        default=5,
        help="Minimum number of frames before confirming an identity (default: 5).",
    )
    parser.add_argument(
        "--max-inactive-frames",
        type=int,
        default=50,
        help="Number of frames after which an inactive ID is removed (default: 50).",
    )

    # Face preprocessing settings
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=80,
        help="Minimum face size (width and height in pixels) for recognition (default: 80).",
    )

    # Output and display settings
    parser.add_argument(
        "--show-fps", action="store_true", help="Overlay FPS counter on the video feed."
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save the output video to disk instead of displaying only.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="output.mp4",
        help="File path to save the output video (only used if --save-video is set).",
    )
    parser.add_argument(
        "--fps-smoothing",
        type=int,
        default=30,
        help="Number of frames over which FPS is averaged (default: 30).",
    )

    # Logging control
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enabled logging (default: disabled).",
    )
    parser.add_argument(
        "--log-destination",
        type=str,
        choices=["terminal", "file", "both"],
        default="terminal",
        help="Where to send log output if logging is enabled: 'terminal', 'file' or 'both' (default: terminal).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        defailt="app.log",
        help="Path to the log file (only used if destination is 'file' or 'both') (default: app.log).",
    )

    return parser.parse_args()
