import cv2
from systems.face_system.face_model.face_model import FaceModel
from systems.face_system.face_bank.face_bank import FaceBank
from systems.face_system.calibrators.yolo_calibrator import YOLOCalibrator
from systems.face_system.calibrators.face_bank_calibrator import FaceBankCalibrator


def main():
    conf, iou = YOLOCalibrator.load_best_thresholds_from_log(
        "logs/yolo_threshold/yolo_2d_sweep_log.txt"
    )
    face_thr = FaceBankCalibrator.load_best_threshold_from_log(
        "logs/face_bank_threshold/threshold_sweep_log.txt"
    )

    print(
        f"[INFO] Loaded thresholds — Conf: {conf:.3f}, IoU: {iou:.3f}, Face: {face_thr:.3f}"
    )

    model = FaceModel(
        detector_weights="assets/weights/YOLOv11/yolov11s_face.pt",
        extractor_arch="edgeface_s_gamma_05",
        extractor_weights="assets/weights/edgeface/edgeface_s_gamma_05.pt",
        device="cuda",
        det_threshold=conf,
        iou_threshold=iou,
    )

    face_bank = FaceBank(
        model=model, bank_path="assets/face_images/", threshold=face_thr
    )
    face_bank.load()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

        cv2.imshow("Live Camera Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
