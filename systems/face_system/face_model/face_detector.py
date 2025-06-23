from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Optional


class FaceDetector:
    """
    YOLO-based face detector that returns bounding boxes for faces in a frame.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        det_threshold: float = 0.5,
        iou_threshold: float = 0.5,
    ):
        """
        Initialize the face detector.

        Args:
            model_path (str): Path to the trained YOLO face detection model.
            device (str): Inference device, e.g., "cuda" or "cpu".
            det_threshold (float): Confidence threshold for filtering detections.
            iou_threshold (float): IoU threshold for NMS during detection.
        """
        self.model = YOLO(model_path, verbose=False)
        self.device = device
        self.det_threshold = det_threshold
        self.iou_threshold = iou_threshold

    def detect(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a BGR frame.

        Args:
            frame (np.ndarray): The original image/frame in BGR.
            conf (Optional[float]): Confidence threshold to filter detections.
                If None, uses the detector's threshold.
        """
        conf = conf if conf is not None else self.det_threshold
        results = self.model.predict(
            frame,
            device=self.device,
            imgsz=320,
            conf=float(conf),
            iou=self.iou_threshold,
            verbose=False,
        )

        boxes = []
        for r in results:
            for box in r.boxes.cpu():
                if box.conf[0] >= conf:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2, y2))

        return boxes
