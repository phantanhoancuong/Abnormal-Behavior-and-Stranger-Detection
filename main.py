from systems.face_system.face_model.face_model import FaceModel
from systems.face_system.calibrators.face_bank_calibrator import FaceBankCalibrator
from systems.face_system.calibrators.yolo_calibrator import YOLOCalibrator


def main():
    yolo_calibrator = YOLOCalibrator(
        model_weights="assets/weights/YOLOv11/yolov11s_face.pt",
        val_images_dir="github/assets/datasets/WIDER Face/WIDER_val/images",
        val_labels_txt=".github/assets/datasets/WIDER Face/wider_face_split/wider_face_val_bbx_gt.txt",
    )
    best_conf, best_iou, best_f1 = yolo_calibrator.sweep_2d(
        output_dir="logs/yolo_threshold/",
        return_result=True,
        load_best_from_log=False,
    )

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
        known_dir="github/assets/datasets/VGGFace2/val",
        unknown_dir="github/assets/datasets/VGGFace2/train",
        bank_path="facebank.npz",
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
        log_path="logs/face_bank_threshold/threshold_sweep_log.txt",
        return_result=True,
    )


if __name__ == "__main__":
    main()
