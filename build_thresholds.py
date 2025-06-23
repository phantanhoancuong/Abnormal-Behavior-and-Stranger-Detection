from systems.face_system.face_model.face_model import FaceModel
from systems.face_system.calibrators.face_bank_calibrator import FaceBankCalibrator
from systems.face_system.calibrators.yolo_calibrator import YOLOCalibrator


def main():
    yolo_calibrator = YOLOCalibrator(
        model_weights="assets/weights/YOLOv11/yolov11s_face.pt",
        val_images_dir="assets/datasets/WIDER Face/WIDER_val/images",
        val_labels_txt="assets/datasets/WIDER Face/wider_face_split/wider_face_val_bbx_gt.txt",
    )

    yolo_result = yolo_calibrator.sweep_2d(
        log_file="logs/yolo_threshold/yolo_2d_sweep_log.txt",
        heatmap_file="logs/yolo_threshold/f1_heatmap.png",
    )

    best_conf = yolo_result["best_conf"]
    best_iou = yolo_result["best_iou"]
    best_f1 = yolo_result["best_f1"]

    model = FaceModel(
        detector_weights="assets/weights/YOLOv11/yolov11s_face.pt",
        extractor_arch="edgeface_s_gamma_05",
        extractor_weights="assets/weights/edgeface/edgeface_s_gamma_05.pt",
        device="cuda",
        det_threshold=best_conf,
        iou_threshold=best_iou,
    )

    face_bank_calibrator = FaceBankCalibrator(
        model=model,
        known_dir="assets/datasets/VGGFace2/val",
        unknown_dir="assets/datasets/VGGFace2/train",
        bank_path="logs/face_bank_threshold/calibration_face_bank.npz",
        max_identities_known=400,
        max_identities_unknown=5000,
        images_per_identity=3,
        probe_images_per_identity=2,
        threshold_start=0.1,
        threshold_end=0.95,
        threshold_step=0.01,
    )

    face_bank_calibrator.build_face_bank()
    face_bank_result = face_bank_calibrator.evaluate(
        log_file="logs/face_bank_threshold/openset_sweep_log.txt",
        plot_file="logs/face_bank_threshold/threshold_plot.png",
    )


if __name__ == "__main__":
    main()
