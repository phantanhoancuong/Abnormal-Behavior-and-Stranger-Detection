import cv2
from systems.face_system.face_model.face_model import FaceModel
from systems.face_system.face_bank.face_bank import FaceBank
from systems.face_system.calibrators.yolo_calibrator import YOLOCalibrator
from systems.face_system.calibrators.face_bank_calibrator import FaceBankCalibrator
from utils.fps_counter import FPSCounter


def main():
    yolo_thresholds = YOLOCalibrator.load_best_thresholds_from_log(
        log_file="logs/yolo_threshold/yolo_2d_sweep_log.txt",
    )
    yolo_conf, yolo_iou = yolo_thresholds["best_conf"], yolo_thresholds["best_iou"]

    face_bank_thresholds = FaceBankCalibrator.load_best_thresholds_from_log(
        "logs/face_bank_threshold/openset_sweep_log.txt"
    )

    face_threshold = face_bank_thresholds["threshold"]

    print(
        f"[INFO] Loaded thresholds — Conf: {yolo_conf:.3f}, IoU: {yolo_iou:.3f}, Face: {face_threshold:.3f}"
    )

    model = FaceModel(
        detector_weights="assets/weights/YOLOv11/yolov11s_face.pt",
        extractor_arch="edgeface_s_gamma_05",
        extractor_weights="assets/weights/edgeface/edgeface_s_gamma_05.pt",
        device="cuda",
        det_threshold=yolo_conf,
        iou_threshold=yolo_iou,
    )

    # Build your own face bank when use
    """ 
    face_bank = FaceBank(
        model=model, bank_path="assets/face_images/", threshold=face_threshold
    )
    face_bank.load()
    """

    # Just testing using the calibration face bank:
    face_bank = FaceBank(
        model=model,
        bank_path="logs/face_bank_threshold/calibration_face_bank.npz",
        threshold=face_threshold,
    )
    face_bank.load()

    fps_counter = FPSCounter(smoothing=20)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fps = fps_counter.update()

        results = model.detect_align_extract(frame)

        for feature, box in results:
            label, score = face_bank.search(feature)
            x1, y1, x2, y2 = box
            color = (0, 255, 0) if label != "stranger" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} ({score:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Live Camera Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
