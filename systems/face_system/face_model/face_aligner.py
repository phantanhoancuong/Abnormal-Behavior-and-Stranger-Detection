import mediapipe as mp
import numpy as np
import cv2
from typing import Tuple, Optional


class FaceAligner:
    """
    Aligns detected face regions using MediaPipe facial landmarks and affine transform.
    """

    def __init__(self, image_size: int = 112):
        """
        Args:
            image_size: Output size of the aligned face image (square).
        """
        self.image_size = image_size
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

        # Landmark indices for: left eye, right eye, nose, left mouth, right mouth
        self.std_indices = [33, 263, 1, 61, 291]

        # Reference standard 5-point landmark positions (for 112x112 output)
        self.std_landmarks = np.array(
            [
                [38.2946, 51.6963],  # left eye
                [73.5318, 51.5014],  # right eye
                [56.0252, 71.7366],  # nose
                [41.5493, 92.3655],  # left mouth
                [70.7299, 92.2041],  # right mouth
            ],
            dtype=np.float32,
        )

    def align(
        self, frame: np.ndarray, face_box: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Aligns the face within the provided bounding box to standard landmark positions.

        Args:
            frame: Original BGR image.
            face_box: Bounding box (x1, y1, x2, y2) of detected face.

        Returns:
            Aligned face image as np.ndarray (shape: [image_size, image_size, 3]),
            or None if alignment fails.
        """
        x1, y1, x2, y2 = face_box
        face_roi = frame[y1:y2, x1:x2]
        h, w, _ = face_roi.shape

        # Run landmark detection
        results = self.mp_face_mesh.process(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None

        # Get 5 landmark points
        landmarks = results.multi_face_landmarks[0]
        pts = []
        for idx in self.std_indices:
            lm = landmarks.landmark[idx]
            pts.append([lm.x * w, lm.y * h])
        pts = np.array(pts, dtype=np.float32)

        # Estimate affine transformation to align to standard positions
        M = cv2.estimateAffinePartial2D(pts, self.std_landmarks)[0]
        aligned_face = cv2.warpAffine(
            face_roi, M, (self.image_size, self.image_size), borderValue=0.0
        )

        return aligned_face
