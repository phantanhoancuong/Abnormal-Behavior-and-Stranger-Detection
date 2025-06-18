import os
import numpy as np
import cv2
import torch
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm


class YOLOCalibrator:
    def __init__(
        self,
        model_weights: str,
        val_images_dir: str,
        val_labels_txt: str,
        label_format: str = "wider",
    ):
        self.model = YOLO(model_weights)
        self.images_dir = val_images_dir
        self.annotations_path = val_labels_txt
        self.label_format = label_format
        self.label_map = self._parse_annotations()

    def _parse_annotations(self) -> Dict[str, List[Tuple[float, float, float, float]]]:
        if self.label_format == "wider":
            return self._parse_wider_annotations()
        elif self.label_format == "yolo":
            return self._parse_yolo_annotations()
        else:
            raise ValueError(f"Unsupported label format: {self.label_format}")

    def _parse_wider_annotations(self):
        label_map = {}
        with open(self.annotations_path, "r") as f:
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

    def _parse_yolo_annotations(self):
        label_map = {}
        for img_file in os.listdir(self.images_dir):
            if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            base_name = os.path.splitext(img_file)[0]
            label_file = os.path.join(self.annotations_path, base_name + ".txt")
            if not os.path.exists(label_file):
                continue
            image_path = os.path.join(self.images_dir, img_file)
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
    def _iou(boxA, boxB) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return inter / (areaA + areaB - inter + 1e-8)

    def sweep_2d(
        self,
        batch_size: int = 8,
        output_dir: str = "log/yolo_threshold/",
        sweep_rounds: int = 2,
        reduction_ratio: float = 0.5,
        conf_start: float = 0.1,
        conf_end: float = 0.8,
        iou_start: float = 0.1,
        iou_end: float = 0.8,
        initial_step: float = 0.1,
        return_result: bool = False,
        load_best_from_log: bool = False,
    ) -> Optional[Dict[str, float]]:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "yolo_2d_sweep_log.txt")
        heatmap_file = os.path.join(output_dir, "f1_heatmap.png")

        if load_best_from_log and os.path.exists(log_file):
            return self._parse_best_from_log(log_file) if return_result else None

        print("[INFO] Performing 2D sweep with adaptive refinement...")
        rel_paths = list(self.label_map.keys())
        detection_cache = {}

        with torch.inference_mode():
            for i in tqdm(range(0, len(rel_paths), batch_size), desc="Running YOLO"):
                batch_paths = rel_paths[i : i + batch_size]
                batch_imgs = [
                    cv2.imread(os.path.join(self.images_dir, p)) for p in batch_paths
                ]
                results = self.model.predict(batch_imgs, conf=0.0, verbose=False)
                for p, res in zip(batch_paths, results):
                    detection_cache[p] = [
                        (tuple(box.xyxy[0].cpu().numpy()), box.conf.item())
                        for box in res.boxes
                    ]

        best_f1 = best_conf = best_iou = 0.0

        def run_grid_sweep(conf_vals, iou_vals, log_f, desc):
            nonlocal best_conf, best_iou, best_f1
            log_f.write(f"# {desc}\n")
            for conf in tqdm(conf_vals, desc=f"{desc} - Conf"):
                for iou in iou_vals:
                    tp = fp = fn = 0
                    for img_path in rel_paths:
                        preds = [
                            (b, s)
                            for b, s in detection_cache.get(img_path, [])
                            if s >= conf
                        ]
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
                        continue
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    log_f.write(f"{conf:.3f}\t{iou:.3f}\t{f1:.4f}\n")

                    if f1 > best_f1:
                        best_f1, best_conf, best_iou = f1, conf, iou

        with open(log_file, "w") as f:
            f.write("CONF\tIOU\tF1\n")
            for round_idx in range(sweep_rounds):
                step = initial_step * (reduction_ratio**round_idx)
                if round_idx == 0:
                    conf_vals = np.arange(conf_start, conf_end + step, step)
                    iou_vals = np.arange(iou_start, iou_end + step, step)
                else:
                    half = step * 3
                    conf_min = max(0.01, best_conf - half)
                    conf_max = min(1.00, best_conf + half)
                    iou_min = max(0.01, best_iou - half)
                    iou_max = min(1.00, best_iou + half)
                    print(
                        f"[Round {round_idx + 1}] Refining: conf=({conf_min:.3f}, {conf_max:.3f}), iou=({iou_min:.3f}, {iou_max:.3f}), step={step:.3f}"
                    )
                    conf_vals = np.arange(conf_min, conf_max + step, step)
                    iou_vals = np.arange(iou_min, iou_max + step, step)

                run_grid_sweep(conf_vals, iou_vals, f, f"Sweep Round {round_idx + 1}")

        self._plot_log_heatmap(log_file, heatmap_file)

        print(
            f"\n[RESULT] Best Conf: {best_conf:.3f}, Best IoU: {best_iou:.3f}, F1: {best_f1:.4f}"
        )
        if return_result:
            return best_conf, best_iou, best_f1
        return None

    def _plot_log_heatmap(self, log_file: str, out_path: str):
        data = np.loadtxt(log_file, skiprows=1, comments="#")
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

    def _parse_best_from_log(self, log_file: str) -> Dict[str, float]:
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
        return {"conf": best_conf, "iou": best_iou, "f1": best_f1}

    @staticmethod
    def load_best_thresholds_from_log(log_path: str) -> Tuple[float, float]:
        data = np.loadtxt(log_path, skiprows=1, comments="#")
        best_idx = np.argmax(data[:, 2])  # F1 score
        return float(data[best_idx, 0]), float(data[best_idx, 1])  # conf, iou
