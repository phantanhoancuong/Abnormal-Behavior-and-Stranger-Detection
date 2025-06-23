import os
import random
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Tuple
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
    """
    Calibrates and sweeps the threshold for open-set face bank search.
    """

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
        """Return IDs (folders) with at least min_images images."""
        return [
            d
            for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
            and len(os.listdir(os.path.join(root, d))) >= min_images
        ]

    def build_face_bank(self) -> None:
        """
        Build and save the face bank from the known_dir.
        """
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
        log_file: Optional[str] = None,
        plot_file: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Sweep thresholds, evaluate on probes, and (optionally) save log/plot.

        Args:
            log_file (str, optional): Where to save sweep log (tab separated, one line per threshold).
            plot_file (str, optional): Where to save the threshold sweep plot.
        """
        y_true, y_scores = [], []

        # Probing known IDs
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

        # Probing strangers
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

        result, log_lines = self._sweep_thresholds(y_true, y_scores)

        if log_file is not None:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(log_file, "w") as f:
                f.write("THRESH\tACC\tPREC\tREC\tF1\tFAR\tFRR\n")
                for line in log_lines:
                    f.write(line)

        if plot_file is not None:
            self._plot_results_from_lines(log_lines, plot_file)

        print("\n[BEST THRESHOLD RESULTS]")
        for k, v in result.items():
            print(f"{k.capitalize()}: {v:.4f}")

        return result

    def _sweep_thresholds(
        self,
        y_true: List[int],
        y_scores: List[float],
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Returns (best_result_dict, log_lines).
        """
        best_acc, best_idx = 0.0, 0
        log_lines = []
        best_stats = {}

        for i, thr in enumerate(self.thresholds):
            y_pred = [int(score >= thr) for score in y_scores]
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            log_lines.append(
                f"{thr:.4f}\t{acc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{far:.4f}\t{frr:.4f}\n"
            )

            if acc > best_acc:
                best_acc, best_idx = acc, i
                best_stats = {
                    "threshold": thr,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "far": far,
                    "frr": frr,
                }

        return best_stats, log_lines

    @staticmethod
    def load_best_thresholds_from_log(log_file: str) -> Dict[str, float]:
        """
        Loads the best threshold/metrics (highest accuracy) from a sweep log file.

        Args:
            log_file (str): Path to the threshold sweep log file.
        """
        data = np.loadtxt(log_file, skiprows=1)
        thresholds = data[:, 0]
        accuracies = data[:, 1]
        best_idx = np.argmax(accuracies)
        # [thresh, acc, prec, rec, f1, far, frr]
        values = data[best_idx]
        return {
            "threshold": float(values[0]),
            "accuracy": float(values[1]),
            "precision": float(values[2]),
            "recall": float(values[3]),
            "f1": float(values[4]),
            "far": float(values[5]),
            "frr": float(values[6]),
        }

    def _plot_results_from_lines(self, log_lines: List[str], plot_file: str) -> None:
        """
        Plot results (accuracy, FAR, FRR) from in-memory log lines and save as PNG.

        Args:
            log_lines (List[str]): Lines in log format (first line is header, skip lines with '#').
            plot_file (str): Where to save the plot.
        """
        data = [
            [float(val) for val in line.strip().split()]
            for line in log_lines
            if not line.startswith("#") and not line.startswith("THRESH")
        ]
        data = np.array(data)
        if data.size == 0:
            return
        thresholds, accs, precs, recs, f1s, fars, frrs = data.T

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
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        plt.savefig(plot_file)
        plt.close()
