from typing import List, Optional, Tuple
import numpy as np

from systems.face_system.face_model.face_detector import FaceDetector
from systems.face_system.face_model.face_aligner import FaceAligner
from systems.face_system.face_model.face_feature_extractor import FaceFeatureExtractor


class FaceModel:
    """
    Face model that handles face detection, alignment, and feature extraction.
    """

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        detector_weights: Optional[str] = None,
        aligner: Optional[FaceAligner] = None,
        extractor: Optional[FaceFeatureExtractor] = None,
        extractor_arch: Optional[str] = None,
        extractor_weights: Optional[str] = None,
        device: str = "cuda",
        det_threshold: float = 0.5,
        iou_threshold: float = 0.5,
    ):
        """
        Initialize the face model with optional custom detector, aligner, and extractor.

        Args:
            detector: Pre-initialized FaceDetector (optional).
            detector_weights: Path to detector weights (required if detector is None).
            aligner: Pre-initialized FaceAligner (optional).
            extractor: Pre-initialized FaceFeatureExtractor (optional).
            extractor_arch: Model architecture name for extractor (if not provided).
            extractor_weights: Path to extractor weights (if not provided).
            device: Device to run models on ('cuda' or 'cpu').
            det_threshold: Default confidence threshold for detection.
            iou_threshold: Default IoU threshold for tracking or matching.
        """
        self.device = device
        self.conf_threshold = det_threshold
        self.iou_threshold = iou_threshold

        if detector is None:
            if detector_weights is None:
                raise ValueError(
                    "detector_weights must be provided if detector is not provided."
                )
            detector = FaceDetector(detector_weights, device)

        if aligner is None:
            aligner = FaceAligner(image_size=112)

        if extractor is None:
            if extractor_arch is None or extractor_weights is None:
                raise ValueError(
                    "Both extractor_arch and extractor_weights must be provided if extractor is not provided."
                )
            extractor = FaceFeatureExtractor(
                extractor_arch=extractor_arch,
                extractor_weights=extractor_weights,
                device=device,
            )

        self.detector = detector
        self.aligner = aligner
        self.extractor = extractor

    def detect_align_extract(
        self, frame: np.ndarray, conf: Optional[float] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect faces, align them, and extract features from a single image.

        Args:
            frame: BGR image as numpy array.
            conf: Detection confidence threshold. Uses default if None.

        Returns:
            List of (embedding, bounding_box) tuples.
        """
        conf = conf if conf is not None else self.conf_threshold
        boxes = self.detector.detect(frame, conf)
        results = []

        for box in boxes:
            aligned_face = self.aligner.align(frame, box)
            if aligned_face is not None:
                feature = self.extractor.extract(aligned_face)
                results.append((feature, box))

        return results

    def detect_align_extract_batch(
        self, frames: List[np.ndarray], conf: Optional[float] = None
    ) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Apply detect_align_extract to each frame in a list individually (naive batching).

        Args:
            frames: List of BGR images (numpy arrays).
            conf: Detection confidence threshold. Uses default if None.

        Returns:
            List of results, one per frame. Each result is a list of (embedding, box) tuples.
        """
        return [self.detect_align_extract(frame, conf) for frame in frames]
