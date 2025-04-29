import cv2
import numpy as np
import torch

REFERENCE_POINTS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

LANDMARK_INDICES = [33, 263, 1, 61, 291]


class FaceProcessor:
    def __init__(
        self, face_mesh, min_face_size, blurry_threshold, det_threshold, target_size=192
    ):
        self.face_mesh = face_mesh
        self.min_face_size = min_face_size
        self.blurry_threshold = blurry_threshold
        self.det_threshold = det_threshold
        self.target_size = target_size

    def is_small_face(self, x1, y1, x2, y2):
        w = x2 - x1
        h = y2 - y1
        return w < self.min_face_size or h < self.min_face_size

    def is_blurry_face(self, face_crop):
        if face_crop.size == 0:
            return True

        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
        return lap_var < self.blurry_threshold

    def pad_to_square(self, image, border_value=0):
        h, w = image.shape[:2]
        size = max(h, w)

        delta_w = size - w
        delta_h = size - h
        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left

        padded_image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=border_value,
        )
        return padded_image

    def align_face(self, face, landmarks, landmark_indices=LANDMARK_INDICES):
        h, w = face.shape[:2]

        coords = np.array(
            [[landmarks[idx].x, landmarks[idx].y] for idx in landmark_indices],
            dtype=np.float32,
        )

        src = coords * np.array([w, h], dtype=np.float32)

        matrix, _ = cv2.estimateAffinePartial2D(src, REFERENCE_POINTS, method=cv2.LMEDS)

        if matrix is None:
            return None

        aligned_face = cv2.warpAffine(face, matrix, (112, 112), borderValue=0)
        return aligned_face

    def prepare_face_tensor(self, face_aligned):
        face_normed = face_aligned.astype(np.float32) / 255.0
        face_normed = (face_normed - 0.5) / 0.5
        face_normed = np.transpose(face_normed, (2, 0, 1))
        face_normed = torch.from_numpy(face_normed)
        return face_normed

    def process_frame(self, frame, detection_result):
        boxes = detection_result.boxes

        if not boxes:
            return None

        aligned_faces = []
        track_ids = []
        bounding_boxes = []

        for box in boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if conf < self.det_threshold:
                continue

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(detection_result.orig_shape[1], x2)
            y2 = min(detection_result.orig_shape[0], y2)

            face_crop = frame[y1:y2, x1:x2]

            face_aligned = self.process_face(face_crop)

            if face_aligned is not None:
                track_id = int(box.id[0]) if box.id is not None else None

                aligned_faces.append(face_aligned)
                track_ids.append(track_id)
                bounding_boxes.append((x1, y1, x2, y2))
        if not aligned_faces:
            return None

        return aligned_faces, track_ids, bounding_boxes

    def process_face(self, face_crop):
        if face_crop.size == 0:
            return None

        if self.is_small_face(0, 0, face_crop.shape[1], face_crop.shape[0]):
            return None

        if self.is_blurry_face(face_crop):
            return None

        face_crop = self.pad_to_square(face_crop)

        h, w = face_crop.shape[:2]
        scale = self.target_size / max(h, w)
        resized_face = cv2.resize(
            face_crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

        face_rgb = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(face_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                aligned_face = self.align_face(resized_face, face_landmarks.landmark)
                if aligned_face is not None:
                    return self.prepare_face_tensor(aligned_face)

        return None
