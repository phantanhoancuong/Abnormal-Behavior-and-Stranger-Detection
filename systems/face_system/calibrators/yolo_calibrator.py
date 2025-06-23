import os
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch


class AnnotationParser:
    @staticmethod
    def parse_wider_annotations(
        path: str,
    ) -> Dict[str, List[Tuple[float, float, float, float]]]:
        """Parses WiderFace-style annotation file."""
        label_map = {}
        with open(path, "r") as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            img_rel_path = lines[i].strip()
            num_faces = int(lines[i + 1].strip())
            boxes = []
            for j in range(num_faces):
                x, y, w, h, *_ = map(float, lines[i + 2 + j].strip().split())
                if w > 0 and h > 0:
                    boxes.append((x, y, x + w, y + h))
            label_map[img_rel_path] = boxes
            i += 2 + num_faces
        return label_map

    @staticmethod
    def parse_yolo_annotations(
        images_dir: str, labels_dir: str
    ) -> Dict[str, List[Tuple[float, float, float, float]]]:
        """Parses YOLO-format label .txt files in a label directory."""
        label_map = {}
        for img_file in os.listdir(images_dir):
            if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            base_name = os.path.splitext(img_file)[0]
            label_file = os.path.join(labels_dir, base_name + ".txt")
            if not os.path.exists(label_file):
                continue
            image_path = os.path.join(images_dir, img_file)
            img = cv2.imread(image_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            boxes = []
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    _, xc, yc, bw, bh = map(float, parts[:5])
                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    x2 = (xc + bw / 2) * w
                    y2 = (yc + bh / 2) * h
                    boxes.append((x1, y1, x2, y2))
            label_map[img_file] = boxes
        return label_map

    @staticmethod
    def parse(
        label_format: str, images_dir: str, labels_path: str
    ) -> Dict[str, List[Tuple[float, float, float, float]]]:
        """Dispatches to the correct annotation parser based on format."""
        if label_format == "wider":
            return AnnotationParser.parse_wider_annotations(labels_path)
        elif label_format == "yolo":
            return AnnotationParser.parse_yolo_annotations(images_dir, labels_path)
        else:
            raise ValueError(f"Unsupported label format: {label_format}")


class YOLOCalibrator:
    """
    Calibrates confidence and IoU thresholds for a YOLO detector.
    """

    def __init__(
        self,
        model_weights: str,
        val_images_dir: str,
        val_labels_txt: str,
        label_format: str = "wider",
        device: Optional[str] = None,
    ):
        """
        Args:
            model_weights (str): Path to YOLO model weights.
            val_images_dir (str): Directory containing validation images.
            val_labels_txt (str): Path to annotation file or YOLO label folder.
            label_format (str): "wider" for WiderFace annotation, "yolo" for YOLO txt format.
            device (Optional[str]): Torch device string, e.g. 'cuda' or 'cpu'.
        """
        self.model = YOLO(model_weights)
        if device is not None:
            self.model.to(device)
        self.images_dir = val_images_dir
        self.annotations_path = val_labels_txt
        self.label_format = label_format
        self.label_map = AnnotationParser.parse(
            label_format=label_format,
            images_dir=val_images_dir,
            labels_path=val_labels_txt,
        )

    @staticmethod
    def _iou(
        boxA: Tuple[float, float, float, float],
        boxB: Tuple[float, float, float, float],
    ) -> float:
        """
        Calculates intersection-over-union for two boxes.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return inter / (areaA + areaB - inter + 1e-8)

    def _read_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")
        return img

    def _get_detection_cache(
        self, rel_paths: List[str], batch_size: int
    ) -> Dict[str, List[Tuple[Tuple[float, float, float, float], float]]]:
        """Run model prediction for all images and cache results."""
        detection_cache = {}
        with torch.inference_mode():
            for i in tqdm(range(0, len(rel_paths), batch_size), desc="Running YOLO"):
                batch_paths = rel_paths[i : i + batch_size]
                batch_imgs = [
                    self._read_image(os.path.join(self.images_dir, p))
                    for p in batch_paths
                ]
                results = self.model.predict(batch_imgs, conf=0.0, verbose=False)
                for p, res in zip(batch_paths, results):
                    detection_cache[p] = [
                        (tuple(box.xyxy[0].cpu().numpy()), box.conf.item())
                        for box in res.boxes
                    ]
        return detection_cache

    def _evaluate_thresholds(
        self,
        rel_paths: List[str],
        detection_cache: Dict[
            str, List[Tuple[Tuple[float, float, float, float], float]]
        ],
        conf: float,
        iou: float,
    ) -> Optional[Tuple[float, float, float]]:
        """Computes precision, recall, F1 for given conf/IoU thresholds."""
        tp = fp = fn = 0
        for img_path in rel_paths:
            preds = [(b, s) for b, s in detection_cache.get(img_path, []) if s >= conf]
            gt_boxes = self.label_map[img_path]
            matched_gt = set()
            for box, _ in preds:
                matched = False
                for i, gt in enumerate(gt_boxes):
                    if i in matched_gt:
                        continue
                    if self._iou(box, gt) >= iou:
                        tp += 1
                        matched_gt.add(i)
                        matched = True
                        break
                if not matched:
                    fp += 1
            fn += len(gt_boxes) - len(matched_gt)
        if tp + fp == 0 or tp + fn == 0:
            return None
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1

    def _grid_search(
        self,
        rel_paths,
        detection_cache,
        sweep_rounds,
        reduction_ratio,
        conf_start,
        conf_end,
        iou_start,
        iou_end,
        initial_step,
        log_lines,
    ):
        """
        Sweeps conf/IoU grid adaptively for multiple rounds and appends log_lines in-memory.
        Consistent round progress/region reporting.
        """
        best_f1 = best_conf = best_iou = 0.0

        def grid_sweep(conf_vals, iou_vals, desc):
            nonlocal best_f1, best_conf, best_iou
            log_lines.append(f"# {desc}\n")
            for conf in tqdm(conf_vals, desc=f"{desc} - Conf"):
                for iou in iou_vals:
                    metrics = self._evaluate_thresholds(
                        rel_paths, detection_cache, conf, iou
                    )
                    if metrics is None:
                        continue
                    precision, recall, f1 = metrics
                    log_lines.append(f"{conf:.3f}\t{iou:.3f}\t{f1:.4f}\n")
                    if f1 > best_f1:
                        best_f1, best_conf, best_iou = f1, conf, iou

        for round_idx in range(sweep_rounds):
            step = initial_step * (reduction_ratio**round_idx)
            if round_idx == 0:
                conf_min, conf_max = conf_start, conf_end
                iou_min, iou_max = iou_start, iou_end
            else:
                half = step * 3
                conf_min = max(0.01, best_conf - half)
                conf_max = min(1.00, best_conf + half)
                iou_min = max(0.01, best_iou - half)
                iou_max = min(1.00, best_iou + half)

            print(
                f"[Round {round_idx + 1}] Sweep: conf=({conf_min:.3f}, {conf_max:.3f}), "
                f"iou=({iou_min:.3f}, {iou_max:.3f}), step={step:.3f}"
            )
            conf_vals = np.arange(conf_min, conf_max + step, step)
            iou_vals = np.arange(iou_min, iou_max + step, step)
            grid_sweep(conf_vals, iou_vals, f"Sweep Round {round_idx + 1}")

        return {
            "best_conf": best_conf,
            "best_iou": best_iou,
            "best_f1": best_f1,
        }

    def sweep_2d(
        self,
        batch_size: int = 8,
        sweep_rounds: int = 2,
        reduction_ratio: float = 0.5,
        conf_start: float = 0.1,
        conf_end: float = 0.8,
        iou_start: float = 0.1,
        iou_end: float = 0.8,
        initial_step: float = 0.1,
        log_file: Optional[str] = None,
        heatmap_file: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Runs the 2D sweep for calibration and (optionally) saves results.

        Args:
            batch_size (int): Number of images per forward pass.
            sweep_rounds (int): Number of refinement rounds.
            reduction_ratio (float): Step size reduction per round.
            conf_start (float): Starting confidence threshold.
            conf_end (float): Ending confidence threshold.
            iou_start (float): Starting IoU threshold.
            iou_end (float): Ending IoU threshold.
            initial_step (float): Initial sweep grid step.
            log_file (Optional[str]): If set, write sweep log to this path.
            heatmap_file (Optional[str]): If set, write heatmap to this path.
        """
        rel_paths = list(self.label_map.keys())
        detection_cache = self._get_detection_cache(rel_paths, batch_size)
        log_lines = []

        best = self._grid_search(
            rel_paths,
            detection_cache,
            sweep_rounds,
            reduction_ratio,
            conf_start,
            conf_end,
            iou_start,
            iou_end,
            initial_step,
            log_lines,
        )

        if log_file is not None:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, "w") as f:
                f.write("CONF\tIOU\tF1\n")
                for line in log_lines:
                    f.write(line)
        if heatmap_file is not None:
            self._plot_log_heatmap_from_lines(log_lines, heatmap_file)

        print(
            f"\n[RESULT] Best Conf: {best['best_conf']:.3f}, Best IoU: {best['best_iou']:.3f}, F1: {best['best_f1']:.4f}"
        )
        return best

    def _plot_log_heatmap_from_lines(self, log_lines: List[str], out_path: str):
        """Plots and saves a heatmap from log lines in memory."""
        data = [
            [float(val) for val in line.strip().split()]
            for line in log_lines
            if not line.startswith("#") and not line.startswith("CONF")
        ]
        data = np.array(data)
        if data.size == 0:
            return
        confs = sorted(set(data[:, 0]))
        ious = sorted(set(data[:, 1]))
        f1_grid = np.zeros((len(confs), len(ious)))
        for row in data:
            c_idx = confs.index(row[0])
            i_idx = ious.index(row[1])
            f1_grid[c_idx, i_idx] = row[2]
        plt.figure(figsize=(8, 6))
        plt.imshow(
            f1_grid,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            extent=(min(ious), max(ious), min(confs), max(confs)),
        )
        plt.colorbar(label="F1 Score")
        plt.xlabel("IoU Threshold")
        plt.ylabel("Confidence Threshold")
        plt.title("YOLO Conf-IoU Threshold Sweep (F1 Heatmap)")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    @staticmethod
    def load_best_thresholds_from_log(
        log_file: str,
    ) -> Dict[str, float]:
        """
        Loads the best result (highest F1) from a sweep log.

        Args:
            log_file (str): Path to the sweep log file.

        """
        best_f1, best_conf, best_iou = -1.0, 0.0, 0.0
        with open(log_file, "r") as f:
            for line in f:
                if line.startswith("#") or line.startswith("CONF"):
                    continue
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                conf, iou, f1 = map(float, parts)
                if f1 > best_f1:
                    best_f1, best_conf, best_iou = f1, conf, iou
        return {"best_conf": best_conf, "best_iou": best_iou, "best_f1": best_f1}
