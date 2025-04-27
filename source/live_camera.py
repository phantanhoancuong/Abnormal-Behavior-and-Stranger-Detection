import cv2
import sys
import pickle
import time
import numpy as np
import argparse

import mediapipe as mp

import torch
import torch.nn.functional as F

from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque


REFERENCE_POINTS = np.array([
    [38.2946, 51.6963],   
    [73.5318, 51.5014],  
    [56.0252, 71.7366],  
    [41.5493, 92.3655],  
    [70.7299, 92.2041]
], dtype = np.float32)

RECOGNIZE_THRESHOLD = 0.5
MIN_FACE_SIZE = 80
BLURRY_THRESHOLD = 60
DETECTION_THRESHOLD = 0.5
MIN_TRACK_AGE = 5
LANDMARK_INDICES = [33, 263, 1, 61, 291]

class FPSCounter:
    """
    A class to compute a smoothed FPS value over recent frames.
    """
    def __init__(self, smoothing = 30):
        """
        Args:
            smoothing (int): Number of recent FPS values to average for smoothing. 
        """
        self.timestamps = deque(maxlen = smoothing)
        self.last_time = None
        

    def update(self):
        """
        Update the FPS counter based on the current frame's processing time.
        Should be called once per frame.
        """    
        current_time = time.time()
        if self.last_time is not None:
            fps = 1.0 / (current_time - self.last_time)
            self.timestamps.append(fps)
        self.last_time = current_time
    
   
    def get_fps(self):
        """
        Get the current smoothed FPS value.
        Returns:
            float: Average FPS over the recent frames.
        """
        if not self.timestamps:
            return 0.0
        return sum(self.timestamps) / len(self.timestamps)


class FaceTracker:
    def __init__(self, recognize_threshold=RECOGNIZE_THRESHOLD):
        """
        Args:
            recognize_threshold (float): Minimum similarity score required to register an identity.
        """
        self.recognize_threshold = recognize_threshold      
        self.id_memory = {}                 # Stores {track_id}: {"identity": name, "similarity": score}}
        self.age_memory = {}                # Stores {track_id: number of frames observed}
        self.last_seen_frame = {}           # Stores {track_id: last frame where it was seen}
        self.global_frame_count = 0         # Counts total processed frames

    def update(self, track_id, identity, similarity):
        """
        Update tracking info for a given track ID.
        Args:
            track_id (int): Unique ID assigned by the object tracker (e.g., ByteTrack).
            identity (str): Predicted identity name or "Stranger".
            similarity (float): Cosine similarity score between face and face bank.
        """
        
        if track_id is None:
            return
        
        # Update track age
        if track_id not in self.age_memory:
            self.age_memory[track_id] = 1
        else:
            self.age_memory[track_id] += 1

        # Update last seen frame
        self.last_seen_frame[track_id] = self.global_frame_count 

        # Update identity if it's new or more confident
        if track_id not in self.id_memory:
            if similarity >= self.recognize_threshold:
                self.id_memory[track_id] = {"identity": identity, "similarity": similarity}
        else:
            if similarity > self.id_memory[track_id]["similarity"]:
                self.id_memory[track_id] = {"identity": identity, "similarity": similarity}

    def get_identity(self, track_id):
        """
        Retrieve the persistent identity for a given track ID.
        Args:
            track_id (int): Track ID to query.
        Returns:
            str: Identity name if known, otherwise "Unknown".
        """
        if track_id is None:
            return "Unknown"
        return self.id_memory.get(track_id, {"identity": "Unknown"})["identity"]

    def get_age(self, track_id):
        """
        Retrieve how many frames a track ID has been alive.
        Args:
            track_id (int): Track ID to query.
        Returns:
            int: Number of frames observed, or 0 if not found.
        """
        return self.age_memory.get(track_id, 0)

    def increment_frame(self):
        """
        Increment the global frame counter.
        Should be called once per frame processed.
        """
        self.global_frame_count += 1

    def cleanup(self, max_inactive_frames = 50):
        """
        Remove old track IDs that have not been seen for a specified number of frames.
        Args:
            max_inactive_frames (int): Maximum number of frames allowed without update before deletion.
        """
        to_delete = []
        for track_id, last_seen in self.last_seen_frame.items():
            if (self.global_frame_count - last_seen) > max_inactive_frames:
                to_delete.append(track_id)

        for track_id in to_delete:
            self.id_memory.pop(track_id, None)
            self.age_memory.pop(track_id, None)
            self.last_seen_frame.pop(track_id, None)

    def reset(self):
        """
        Completely reset the tracker memory and frame counter.
        """
        self.id_memory.clear()
        self.age_memory.clear()
        self.last_seen_frame.clear()
        self.global_frame_count = 0

def is_small_face(x1, y1, x2, y2, min_size):
    """
    Args:
        x1, y1 (int): Top-left corner of the bounding box.
        x2, y2 (int): Bottom-right corner of the bounding box.
        min_size (int): Minimum acceptable width/height for a face.
    Returns:
        bool: True if the face is too small, False otherwise.
    """
    w = x2 - x1
    h = y2 - y1
    return w < min_size or h < min_size

def is_blurry_face(image, blurry_threshold = BLURRY_THRESHOLD):
    """
    Check if the cropped face image is too blurry based on Laplacian variance.
    Args:
        image (np.ndarray): Cropped face region in BGR format.
        blurry_threshold (float): Variance threshold below which image is considered blurry.
    Returns:
        bool: True if the image is blurry or invalid, False otherwise.
    """
    if image.size == 0:
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < blurry_threshold    

def load_face_bank(face_bank_path):
    """
    Load the saved face bank (pre-computed face embeddings) from disk.
    Args:
        face_bank_path (str): Path to the serialized face bank file (.pkl).
    Returns:
        dict: Dictionary mapping identity names to lists of embeddings.
    Raises:
        SystemExit: If loading fails (file not found, corrupt, etc).
    """
    try:
        with open(face_bank_path, 'rb') as f:
            face_bank = pickle.load(f)
        print(f"[INFO] Loaded face bank with {len(face_bank)} identities.")
        return face_bank
    except Exception as e:
        print(f"[ERROR] Failed to load face bank: {e}")
        sys.exit(1)

def recognize_face(embedding, face_bank, threshold = RECOGNIZE_THRESHOLD):
    """
    Compare a face embedding against a face bank to recognize identity.
    Args:
        embedding (np.ndarray): The embedding vector of the detected face (shape: [embedding_dim]).
        face_bank (dict): A dictionary mapping identity names to lists of embeddings.
        threshold (float): Similarity threshold to accept recognition; otherwise classified as "Stranger".
    Returns:
        tuple:
            - identity (str): Recognized identity name, or "Stranger" if below threshold.
            - max_sim (float): The highest cosine similarity score found.
    """
    max_sim = -1
    identity = "Stranger"
    for name, emb_list in face_bank.items():
        sims = cosine_similarity([embedding], emb_list)[0]
        best_sim = np.max(sims)
        if best_sim > max_sim:
            max_sim = best_sim
            identity = name
    if max_sim < threshold:
        identity = "Stranger"
    return identity, max_sim 

def preprocess_face(face_aligned):
    """
    Preprocess an aligned face image for embedding extraction.
    Steps:
    - Normalize pixel values to [-1, 1] range.
    - Rearrange dimensions to channel-first format (C, H, W).
    - Convert to a PyTorch tensor.
    Args:
        face_aligned (np.ndarray): Aligned face image in (H, W, C) format, BGR or RGB.
    Returns:
        torch.Tensor: Preprocessed face tensor ready for model input (shape: [3, 112, 112]).
    """
    face = face_aligned.astype(np.float32) / 255.0        # Scale to [0, 1]
    face = (face - 0.5) / 0.5                             # Normalize to [-1, 1]
    face = np.transpose(face, (2, 0, 1))                  # Change to (C, H, W)
    face = torch.from_numpy(face)                         # Convert to PyTorch tensor
    return face
    


def align_face(face, landmarks, landmark_indices = LANDMARK_INDICES):
    """
    Align a detected face image based on facial landmarks.
    Args:
        face (np.ndarray): Cropped face image (BGR or RGB).
        landmarks (List[Landmark]): List of 468 MediaPipe landmarks.
        landmark_indices (List[int]): Indices of 5 key landmarks used for alignment.
    Returns:
        np.ndarray or None:
            - Aligned face image of size (112, 112) if successful.
            - None if alignment matrix estimation fails.
    """
    h, w = face.shape[:2]

    coords = np.array(
        [[landmarks[idx].x, landmarks[idx].y] for idx in landmark_indices],
        dtype=np.float32
    )
    src = coords * np.array([w, h], dtype = np.float32)

    matrix, _ = cv2.estimateAffinePartial2D(src, REFERENCE_POINTS, method = cv2.LMEDS)

    if matrix is None:
        return None 

    aligned_face = cv2.warpAffine(face, matrix, (112, 112), borderValue = 0)
    return aligned_face

def process_single_face(face_roi, face_mesh, min_face_size, target_size = 192):
    if face_roi.size == 0 or is_small_face(0, 0, face_roi.shape[1], face_roi.shape[0], min_face_size) or is_blurry_face(face_roi):
        return None
    
    h, w = face_roi.shape[:2]
    scale = target_size / max(h, w)
    resized_w, resized_h = int(w * scale), int(h * scale)
    resized_face = cv2.resize(face_roi, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    
    
    face_rgb = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
    results_mp = face_mesh.process(face_rgb)
    
    if results_mp.multi_face_landmarks:
        for face_landmarks in results_mp.multi_face_landmarks:
            aligned_face = align_face(face_roi, face_landmarks.landmark)
            if aligned_face is not None:
                return preprocess_face(aligned_face)
            
    return None

def draw_face_info(frame, track_id, identity, similarity, box):
    x1, y1, x2, y2 = box
    color = (0, 0, 255) if identity == "Stranger" else(0, 255, 0)
    label = f"ID: {track_id} | Identity: {identity} | Sim: {similarity:.2f}"
    
    (text_width, _), _ = cv2.getTextSize(text = label, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.8, thickness = 2)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    center_x = (x1 + x2) // 2
    text_x = center_x - (text_width // 2)
    text_y = max(0, y1 - 10)
    
    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    
def parse_args():
    parser = argparse.ArgumentParser(description = "Real-time Face Detection and Recognition")    
    
    parser.add_argument(
        "--mode", "-m",
        choices = ["amp", "no_amp"],
        default = "no_amp",
        help = "Use 'amp' for Automatic Mixed-Precision or 'no_amp for standard mode (default: no_amp)."
    )
    parser.add_argument(
        "--source", "-s",
        type = str,
        default = "0",
        help = "Camera index (e.g., 0) or path to a video file (default: 0)."
    )
    parser.add_argument(
        "--detector", "-d",
        type = str,
        default = "models/face-detector/YOLOv11/yolov11s-face.pt",
        help = "Path to YOLO face detector weights (default: models/face-detector/YOLOv11/yolov11s-face.pt)."
    )
    parser.add_argument(
        "--tracker", "-t",
        type = str,
        default = "models/configs/bytetrack.yaml",
        help = "Tracker config file (default: models/configs/bytetrack.yaml)"
    )
    parser.add_argument(
        "--face-bank", "-f",
        type = str,
        default = "data/face-bank.pkl",
        help = "Path to face bank pickle file (.pkl) (default: data/face_banks/face-bank.pkl)."
    )
    parser.add_argument(
        "--min-track-age",
        type = int,
        default = 5,
        help = "Minimum frames before confirming an ID (default: 5)."
    )
    parser.add_argument(
        "--min-face-size",
        type = int,
        default = 80,
        help = "Minimum number of pixels for width and height of a face in pixels (default: 80)."
    )
    parser.add_argument(
        "--det-threshold",
        type = float,
        default = 0.5,
        help = "Face detection confidence threshold (default = 0.5)"
    )
    parser.add_argument(
        "--rec-threshold",
        type = float,
        default = 0.5,
        help = "Face recognition confidence threshold (default = 0.5)"
    )
    parser.add_argument(
        "--show-fps",
        action = "store_true",
        help = "If set, overlay the FPS counter on the video."
    )
    parser.add_argument(
        "--show-box",
        action = "store_true",
        help = "If set, draw face bounding boxes even if identity is unknown."
    )
    parser.add_argument(
        "--max-inactive-frames",
        type = int,
        default = 50,
        help = "Number of frames after which an inactive ID is removed from memory (default: 50)."
    )
    parser.add_argument(
        "--save-video",
        action = "store_true",
        help = "If set, saves the output video to disk instead of only showing it live."
    )
    parser.add_argument(
        "--output-path",
        type = str,
        default = "output.mp4",
        help = "Path where the output video will be saved (only if --save-video is used) (default: output.mp4)."
    )

    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"[INFO] torch.cuda.is_available() = {torch.cuda.is_available()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    amp_enabled = False
    
    if args.mode == "amp":
        if device.type == 'cuda':
            print(f"[INFO] CUDA system used, AMP enabled.")
            amp_enabled = True
        else:
            print(f"[WARNING] Non-CUDA system used, AMP disabled.")
    
    face_bank = load_face_bank(args.face_bank)
    detector = YOLO(args.detector, verbose = False)
    
    model = torch.hub.load(
        repo_or_dir = 'otroshi/edgeface',
        model = 'edgeface_s_gamma_05',
        source = 'github',
        pretrained = True
    ).to(device)
    model.eval()
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode = False, refine_landmarks = True)
    
    face_tracker = FaceTracker(recognize_threshold = args.rec_threshold)
    
    source = int(args.source) if args.source.isdigit() else args.source
    
    results = detector.track(
        source = source,
        stream = True,
        persist = True,
        tracker = args.tracker,
    )
    
    fps_counter = FPSCounter(smoothing = 30)

    out = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    try:
        for result in results:
            fps_counter.update()
            face_tracker.increment_frame()
            
            frame = result.orig_img
            if out is None and args.save_video:
                h, w = frame.shape[:2]
                out = cv2.VideoWriter(args.output_path, fourcc, 30, (w, h))
            
            boxes = result.boxes
                        
            if not boxes:
                continue
            
            aligned_faces = []
            track_ids = []
            boxes_xyxy = []
            
            for box in boxes:
                track_id = int(box.id[0]) if box.id is not None else None
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                if conf < args.det_threshold:
                    continue
                
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(result.orig_shape[1], x2)
                y2 = min(result.orig_shape[0], y2)
                
                face_roi = frame[y1:y2, x1:x2]
                aligned = process_single_face(face_roi, face_mesh, min_face_size = args.min_face_size)
                
                if aligned is not None:
                    aligned_faces.append(aligned)
                    track_ids.append(track_id)
                    boxes_xyxy.append((x1, y1, x2, y2))
                    
            if aligned_faces:
                batch = torch.stack(aligned_faces).to(device)
                
                if amp_enabled:
                    batch = batch.half()
                    
                with torch.autocast(device_type = device.type, enabled = amp_enabled):
                    embeddings = model(batch)
                    embeddings = F.normalize(embeddings, p = 2, dim = 1)
                    
                embeddings = embeddings.detach().float().cpu().numpy()
                
                for emb, tid, (x1, y1, x2, y2) in zip(embeddings, track_ids, boxes_xyxy):
                    identity, sim = recognize_face(emb, face_bank)
                    face_tracker.update(tid, identity, sim)
                    
                    if face_tracker.get_age(tid) < args.min_track_age:
                        continue
                    
                    persistent_identity = face_tracker.get_identity(tid)
                    
                    if args.show_box or persistent_identity != "Stranger":
                        draw_face_info(
                            frame = frame,
                            track_id = tid,
                            identity = persistent_identity,
                            similarity = sim,
                            box =  (x1, y1, x2, y2)
                        )
                        
            if args.show_fps:
                fps_text = f"FPS: {fps_counter.get_fps():.1f}"
                cv2.putText(frame, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            
            if args.save_video and out is not None:
                out.write(frame)
                
            
            
            cv2.imshow("Face Detection + Recognition", frame)
            face_tracker.cleanup(max_inactive_frames = args.max_inactive_frames)
            
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        if args.save_video and out is not None:
            out.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        detector.close()
        sys.exit(0)
        
if __name__ == "__main__":
    main()