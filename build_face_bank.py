import os
import cv2
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
import argparse

from ultralytics import YOLO
from collections import defaultdict

REFERENCE_POINTS = np.array([
    [38.2946, 51.6963],   
    [73.5318, 51.5014],  
    [56.0252, 71.7366],  
    [41.5493, 92.3655],  
    [70.7299, 92.2041]
], dtype=np.float32)

LANDMARK_INDICES = [33, 263, 1, 61, 291]

def preprocess_face(face_aligned):
    face = face_aligned.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5
    face = np.transpose(face, (2, 0, 1))
    return torch.from_numpy(face)

def align_face(face, landmarks, landmark_indices=LANDMARK_INDICES):
    h, w = face.shape[:2]
    coords = np.array(
        [[landmarks[idx].x, landmarks[idx].y] for idx in landmark_indices],
        dtype=np.float32
    )
    src = coords * np.array([w, h], dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(src, REFERENCE_POINTS, method=cv2.LMEDS)
    if matrix is None:
        return None
    aligned_face = cv2.warpAffine(face, matrix, (112, 112), borderValue=0)
    return aligned_face

def is_blurry(image, threshold=60):
    if image.size == 0:
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold

def extract_face_embedding(image, detector, face_mesh, model, device):
    results = detector.predict(image)
    if not results or not results[0].boxes:
        return None
    
    box = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    if conf < 0.5:
        return None

    face_roi = image[y1:y2, x1:x2]
    if face_roi.size == 0 or is_blurry(face_roi):
        return None

    h, w = face_roi.shape[:2]
    scale = 192 / max(h, w)
    resized_face = cv2.resize(face_roi, (int(w * scale), int(h * scale)))

    target_size = 192
    h, w = resized_face.shape[:2]
    top = (target_size - h) // 2
    bottom = target_size - h - top
    left = (target_size - w) // 2
    right = target_size - w - left
    padded_face = cv2.copyMakeBorder(resized_face, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])

    face_rgb = cv2.cvtColor(padded_face, cv2.COLOR_BGR2RGB)
    result_mesh = face_mesh.process(face_rgb)

    if result_mesh.multi_face_landmarks is None:
        return None

    aligned = align_face(padded_face, result_mesh.multi_face_landmarks[0].landmark)
    if aligned is None:
        return None

    face_tensor = preprocess_face(aligned).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(face_tensor)
        emb = F.normalize(emb, p=2, dim=1)
    return emb.squeeze(0).cpu().numpy()

def build_face_bank(image_folder, output_path, device):
    detector = YOLO('/models/face-detector/YOLOv11/yolov11s-face.pt', verbose=False)
    model = torch.hub.load('otroshi/edgeface', 'edgeface_s_gamma_05', source='github', pretrained=True).to(device)
    model.eval()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

    face_bank = defaultdict(list)

    for filename in os.listdir(image_folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        filepath = os.path.join(image_folder, filename)
        identity = filename.split('_')[0]  

        img = cv2.imread(filepath)
        if img is None:
            print(f"[WARNING] Failed to read {filename}. Skipping.")
            continue

        emb = extract_face_embedding(img, detector, face_mesh, model, device)
        if emb is not None:
            face_bank[identity].append(emb)
            print(f"[INFO] Added embedding for {identity}: {filename}")
        else:
            print(f"[WARNING] Failed to extract embedding from {filename}")

    with open(output_path, 'wb') as f:
        pickle.dump(dict(face_bank), f)
    print(f"[SUCCESS] Face bank saved to {output_path}")

    face_mesh.close()

def main():
    parser = argparse.ArgumentParser(description="Build face bank from images")
    parser.add_argument("--image-folder", type=str, required=True, help="Path to folder with face images")
    parser.add_argument("--output-path", type=str, default="data/face-bank.pkl", help="Path to save face bank .pkl file")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    build_face_bank(args.image_folder, args.output_path, device)

if __name__ == "__main__":
    main()
