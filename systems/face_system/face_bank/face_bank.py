import os
import cv2
import numpy as np
import faiss
from typing import List, Tuple, Union


class FaceBank:
    """
    FaceBank class for storing, managing, and querying face embeddings using FAISS for fast similarity search.
    """

    def __init__(self, model, bank_path: str, threshold: float = 0.15):
        """
        Initialize the FaceBank instance.

        Args:
            model: A face recognition model with method `detect_align_extract(image)` → List[Tuple[embedding, ...]]
            bank_path: Path to store the facebank `.npz` file.
            threshold: Cosine similarity threshold to determine if a match is a known identity.
        """
        self.model = model
        self.bank_path = bank_path
        self.threshold = threshold
        self.embeddings: List[np.ndarray] = []
        self.labels: List[str] = []
        self.index: Union[faiss.IndexFlatIP, None] = None

    def build(self, face_bank_dir: str) -> None:
        """
        Clear, populate, save, and load the face bank from a given directory.

        Args:
            face_bank_dir: Root directory containing subdirectories for each person with face images.
        """
        self.clear()
        self.append(face_bank_dir)
        self.save()
        self.load()

    def append(self, face_bank_dir: str) -> None:
        for person in sorted(os.listdir(face_bank_dir)):
            person_path = os.path.join(face_bank_dir, person)
            if not os.path.isdir(person_path):
                continue

            image_paths = []
            for image_file in sorted(os.listdir(person_path)):
                image_path = os.path.join(person_path, image_file)
                image_paths.append(image_path)

            # Read and store all images
            images = [cv2.imread(path) for path in image_paths]
            images = [img for img in images if img is not None]

            if not images:
                continue

            # Use batch inference
            detections_batch = self.model.detect_align_extract_batch(images)
            count = 0

            for detections in detections_batch:
                if not detections:
                    continue
                feature, _ = detections[0]
                self.embeddings.append(feature.astype(np.float32))
                self.labels.append(person)
                count += 1

            print(f"[INFO] Added {count} image(s) for {person}")

    def clear(self) -> None:
        """
        Clear the current face bank embeddings and labels and delete the .npz file if exists.
        """
        self.embeddings = []
        self.labels = []
        self.index = None

        if os.path.exists(self.bank_path):
            os.remove(self.bank_path)
            print(f"[INFO] Cleared existing facebank at {self.bank_path}")

    def save(self) -> None:
        """
        Save the current face bank (embeddings and labels) to a compressed `.npz` file.
        """
        dir_path = os.path.dirname(self.bank_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        np.savez(
            self.bank_path,
            embeddings=np.stack(self.embeddings, axis=0).astype(np.float32),
            labels=np.array(self.labels),
        )
        print(f"[INFO] Face bank saved to {self.bank_path}")

    def load(self) -> None:
        """
        Load embeddings and labels from file and build a FAISS index for cosine similarity search.
        """
        if not os.path.exists(self.bank_path):
            raise FileNotFoundError(
                f"[ERROR] Face bank file not found: {self.bank_path}"
            )

        data = np.load(self.bank_path)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.labels = data["labels"].tolist()
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, feature: np.ndarray, topk: int = 1) -> Tuple[str, float]:
        """
        Search the face bank for the most similar identity given a query feature.

        Args:
            feature: 1D L2-normalized face embedding (e.g., shape [512]).
            topk: Number of top results to retrieve (default 1).

        Returns:
            Tuple[str, float]: (label or "stranger", cosine similarity score)
        """
        feature = feature.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(feature, topk)

        score = scores[0][0]
        idx = indices[0][0]
        label = self.labels[idx]

        if score >= self.threshold:
            return label, score
        else:
            return "stranger", score

    def __len__(self) -> int:
        return len(self.labels)

    def __contains__(self, label: str) -> bool:
        return label in self.labels
