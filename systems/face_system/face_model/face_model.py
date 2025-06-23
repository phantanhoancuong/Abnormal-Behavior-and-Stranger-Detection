from typing import List, Optional, Tuple
import numpy as np
import torch

from systems.face_system.face_model.face_detector import FaceDetector
from systems.face_system.face_model.face_aligner import FaceAligner
from systems.face_system.face_model.face_feature_extractor import FaceFeatureExtractor


class FaceModel:
    """
    Face system pipeline that handles face detection, alignment, and feature extraction.
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
        Initialize the face pipeline.

        Args:
            detector (Optional[FaceDetector]): Pre-initialized FaceDetector instance (optional).
            detector_weights (Optional[str]): Path to detector weights (required if detector is None).
            aligner (Optional[FaceAligner]): Pre-initialized FaceAligner instance (optional).
            extractor (Optional[FaceFeatureExtractor]): Pre-initialized FaceFeatureExtractor instance (optional).
            extractor_arch (Optional[str]): Architecture name for extractor (required if extractor is None).
            extractor_weights (Optional[str]): Path to extractor weights (required if extractor is None).
            device (str): Device string ("cuda" or "cpu").
            det_threshold (float): Confidence threshold for detection.
            iou_threshold (float): IoU threshold for detection/postprocessing.

        Raises:
            ValueError: If required weights/architecture are missing for initialization.
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
        Detect faces, align them, and extract feature embeddings from a single frame.

        Args:
            frame (np.ndarray): The original image/frame in BGR.
            conf (Optional[float]): Detection confidence threshold (optional).
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
        Detect faces, align them, and batch-extract features for all frames.

        Args:
            frames (List[np.ndarray]): List of BGR images.
            conf (Optional[float]): Detection confidence threshold.
        """
        conf = conf if conf is not None else self.conf_threshold
        all_aligned = []
        all_boxes = []
        frame_indices = []

        for i, frame in enumerate(frame):
            boxes = self.detector.detect(frame, conf)
            for box in boxes:
                aligned = self.aligner.align(frame, box)
                if aligned is not None:
                    preprocessed = self.extractor.preprocess(aligned)
                    all_aligned.append(preprocessed)
                    all_boxes.append(box)
                    frame_indices.append(i)

        if not all_aligned:
            return [[] for _ in frames]

        batch_tensor = (
            torch.from_numpy(np.stack(all_aligned)).float().to(self.extractor.device)
        )
        features = self.extractor.forward(batch_tensor)

        results = [[] for _ in frames]
        for idx, (feature, box) in zip(frame_indices, zip(features, all_boxes)):
            results[idx].append((feature, box))

        return results
