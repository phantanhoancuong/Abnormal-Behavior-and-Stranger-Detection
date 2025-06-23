import mediapipe as mp
import numpy as np
import cv2
from typing import Tuple, Optional


class FaceAligner:
    """
    Using MediaPipe facial landmarks to align face ROIs to a standardized five-point reference (eyes, nose, mouth corners).
    Outputting square-aligned faces for recognition models.
    """

    def __init__(self, image_size: int = 112):
        """
        Initialize FaceAligner with the desired output size.

        Args:
            image_size (int): Output width/height for the aligned face image (in pixels).
        """
        self.image_size = image_size
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

        self.std_indices = [33, 263, 1, 61, 291]

        self.std_landmarks = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )

    def align(
        self, frame: np.ndarray, face_box: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Align a face from the input frame using detected bounding box and MediaPipe facial landmarks.

        Args:
            frame (np.ndarray): The original image/frame in BGR.
            face_box (Tuple[int, int, int, int]): The bounding box for the detected face as (x1, y1, x2, y2).
                - (x1, y1): top-left corner; (x2, y2): bottom-right corner. Should be within image bounds.
        """
        x1, y1, x2, y2 = map(int, face_box)
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w_img), min(y2, h_img)
        if x2 <= x1 or y2 <= y1:
            return None

        face_roi = frame[y1:y2, x1:x2]
        h, w = face_roi.shape[:2]
        if h == 0 or w == 0:
            return None

        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(face_rgb)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        pts = np.array(
            [
                [landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h]
                for idx in self.std_indices
            ],
            dtype=np.float32,
        )

        M = cv2.estimateAffinePartial2D(pts, self.std_landmarks)[0]
        if M is None:
            return None

        aligned_face = cv2.warpAffine(
            face_roi, M, (self.image_size, self.image_size), borderValue=0
        )
        return aligned_face
