import os
import random
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from systems.face_system.face_bank.face_bank import FaceBank
from systems.face_system.face_model.face_model import FaceModel


class FaceBankCalibrator:
    def __init__(
        self,
        model: FaceModel,
        known_dir: str,
        unknown_dir: str,
        bank_path: str,
        max_identities_known: int,
        max_identities_unknown: int,
        images_per_identity: int,
        probe_images_per_identity: int = 3,
        threshold_start: float = 0.0,
        threshold_end: float = 1.0,
        threshold_step: float = 0.01,
    ):
        self.model = model
        self.known_dir = known_dir
        self.unknown_dir = unknown_dir
        self.bank_path = bank_path
        self.max_identities_known = max_identities_known
        self.max_identities_unknown = max_identities_unknown
        self.images_per_identity = images_per_identity
        self.probe_images_per_identity = probe_images_per_identity

        assert 0.0 <= threshold_start < threshold_end <= 1.0
        assert 0.0 < threshold_step < (threshold_end - threshold_start)

        self.thresholds = np.arange(
            threshold_start, threshold_end + 1e-6, threshold_step
        )
        self.face_bank = FaceBank(model=model, bank_path=bank_path)

    def _get_valid_ids(self, root: str, min_images: int) -> List[str]:
        return [
            d
            for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
            and len(os.listdir(os.path.join(root, d))) >= min_images
        ]

    def build_face_bank(self) -> None:
        min_images = self.images_per_identity + self.probe_images_per_identity
        all_ids = self._get_valid_ids(self.known_dir, min_images)
        selected_ids = random.sample(
            all_ids, min(self.max_identities_known, len(all_ids))
        )

        self.face_bank.clear()

        for identity in tqdm(selected_ids, desc="Building FaceBank"):
            person_path = os.path.join(self.known_dir, identity)
            image_files = sorted(os.listdir(person_path))[: self.images_per_identity]
            for img_file in image_files:
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                detections = self.model.detect_align_extract(img)
                if not detections:
                    continue
                feature, _ = detections[0]
                self.face_bank.embeddings.append(feature.astype(np.float32))
                self.face_bank.labels.append(identity)

        self.face_bank.save()
        self.face_bank.load()

    def evaluate(
        self,
        log_path: Optional[str] = "log/face_bank_threshold/openset_sweep_log.txt",
        return_result: bool = False,
    ) -> Optional[Dict[str, float]]:
        y_true, y_scores = [], []

        for identity in tqdm(self.face_bank.labels, desc="Probing knowns"):
            person_path = os.path.join(self.known_dir, identity)
            image_files = sorted(os.listdir(person_path))[
                self.images_per_identity : self.images_per_identity
                + self.probe_images_per_identity
            ]
            for img_file in image_files:
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                detections = self.model.detect_align_extract(img)
                if not detections:
                    continue
                feature, _ = detections[0]
                _, score = self.face_bank.search(feature)
                y_true.append(1)
                y_scores.append(score)

        stranger_ids = self._get_valid_ids(
            self.unknown_dir, self.probe_images_per_identity
        )
        selected_strangers = random.sample(
            stranger_ids, min(self.max_identities_unknown, len(stranger_ids))
        )

        for identity in tqdm(selected_strangers, desc="Probing strangers"):
            person_path = os.path.join(self.unknown_dir, identity)
            image_files = sorted(os.listdir(person_path))[
                : self.probe_images_per_identity
            ]
            for img_file in image_files:
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                detections = self.model.detect_align_extract(img)
                if not detections:
                    continue
                feature, _ = detections[0]
                _, score = self.face_bank.search(feature)
                y_true.append(0)
                y_scores.append(score)

        return self._sweep_thresholds(y_true, y_scores, log_path, return_result)

    def _sweep_thresholds(
        self,
        y_true: List[int],
        y_scores: List[float],
        log_path: Optional[str],
        return_result: bool,
    ) -> Optional[Dict[str, float]]:
        best_acc, best_idx = 0.0, 0

        if log_path:
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            log_f = open(log_path, "w")
            log_f.write("THRESH\tACC\tPREC\tREC\tF1\tFAR\tFRR\n")
        else:
            log_f = None

        for i, thr in enumerate(self.thresholds):
            y_pred = [int(score >= thr) for score in y_scores]
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            if acc > best_acc:
                best_acc, best_idx = acc, i

            if log_f:
                log_f.write(
                    f"{thr:.4f}\t{acc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{far:.4f}\t{frr:.4f}\n"
                )

        if log_f:
            log_f.close()

        self._plot_results(log_path)

        result = {
            "threshold": self.thresholds[best_idx],
            "accuracy": best_acc,
            "precision": precision_score(
                y_true, [int(s >= self.thresholds[best_idx]) for s in y_scores]
            ),
            "recall": recall_score(
                y_true, [int(s >= self.thresholds[best_idx]) for s in y_scores]
            ),
            "f1": f1_score(
                y_true, [int(s >= self.thresholds[best_idx]) for s in y_scores]
            ),
            "far": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            "frr": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        }

        print("\n[BEST THRESHOLD RESULTS]")
        for k, v in result.items():
            print(f"{k.capitalize()}: {v:.4f}")

        return result if return_result else None

    @staticmethod
    def load_best_threshold_from_log(log_path: str) -> float:
        """
        Loads the best threshold (with highest accuracy) from the sweep log file.

        Args:
            log_path (str): Path to the threshold sweep log file.

        Returns:
            float: Best threshold value.
        """
        data = np.loadtxt(log_path, skiprows=1)
        thresholds = data[:, 0]
        accuracies = data[:, 1]
        best_idx = np.argmax(accuracies)
        return float(thresholds[best_idx])

    def _plot_results(self, log_path: Optional[str]) -> None:
        if log_path is None or not os.path.exists(log_path):
            return
        data = np.loadtxt(log_path, skiprows=1)
        thresholds, accs, fars, frrs = data[:, 0], data[:, 1], data[:, 5], data[:, 6]

        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, accs, label="Accuracy")
        plt.plot(thresholds, fars, label="FAR")
        plt.plot(thresholds, frrs, label="FRR")
        plt.xlabel("Threshold")
        plt.ylabel("Metric Value")
        plt.title("Face Bank Threshold Calibration")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        outpath = os.path.join(os.path.dirname(log_path), "threshold_plot.png")
        plt.savefig(outpath)
        plt.close()
