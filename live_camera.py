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
    def __init__(self, recognize_threshold):
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

def is_small_face(x1, y1, x2, y2, min_face_size):
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
    return w < min_face_size or h < min_face_size

def is_blurry_face(image, blurry_threshold):
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

def match_face_embeddings(embedding, face_bank, threshold):
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

def prepare_face_tensor(face_aligned):
    """
    Normalize an aligned face image and convert it into a PyTorch tensor.
    Processing steps:
    - Scale pixel values from [0, 255] to [0, 1].
    - Normalize pixel values to the range [-1, 1].
    - Rearrange dimensions from (Height, Width, Channels) to (Channels, Height, Width).
    - Convert the NumPy array into a PyTorch tensor.
    Args:
        face_aligned (np.ndarray): Input aligned face image in (H, W, C) format, either BGR or RGB.
    Returns:
        torch.Tensor: A normalized face tensor of shape (3, 112, 112), ready for model input.
    """
    face = face_aligned.astype(np.float32) / 255.0  # Scale pixel values to [0, 1]
    face = (face - 0.5) / 0.5                       # Normalize to [-1, 1]
    face = np.transpose(face, (2, 0, 1))            # Change layout to (Channels, Height, Width)
    face = torch.from_numpy(face)                   # Convert to PyTorch tensor
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

def pad_to_square(image, border_value = 0):
    """
    Pad an image to make it square by adding borders.
    Args:
        image (np.ndarray): Input image of shape (H, W, C) or (H, W).
        border_value (int or tuple, optional): Color value for the padding. Default is 0 (black).
    Returns:
        np.ndarray: Padded square image.
    """
    h, w = image.shape[:2]
    size = max(h, w)

    # Padding amounts
    delta_w = size - w
    delta_h = size - h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    padded_image = cv2.copyMakeBorder(
        image, top, bottom, left, right, 
        borderType=cv2.BORDER_CONSTANT, value=border_value
    )
    return padded_image

def process_single_face(face_crop, face_mesh, min_face_size, blurry_threshold, target_size = 192):
    """
    Preprocess a single cropped face for face recognition:
    - Validate size and sharpness.
    - Pad to square.
    - Resize to target size.
    - Detect facial landmarks.
    - Align based on landmarks.
    - Normalize to a tensor.
    
    Args:
        face_crop (np.ndarray): Cropped face image (BGR).
        face_mesh (mp.solutions.face_mesh.FaceMesh): MediaPipe face landmark detector.
        min_face_size (int): Minimum acceptable face dimension (width/height).
        target_size (int, optional): Target maximum size for the longest side before landmarking. Default is 192.
    
    Returns:
        torch.Tensor or None: Preprocessed face tensor (shape [3, 112, 112]), or None if preprocessing fails.
    """

    if face_crop.size == 0:
        return None
    if is_small_face(0, 0, face_crop.shape[1], face_crop.shape[0], min_face_size):
        return None
    if is_blurry_face(face_crop, blurry_threshold = blurry_threshold):
        return None

    # Pad to square shape for MediaPipe
    face_crop = pad_to_square(face_crop)
    
    h, w = face_crop.shape[:2]
    scale = target_size / max(h, w)
    resized_face = cv2.resize(face_crop, (int(w * scale), int(h * scale)), interpolation = cv2.INTER_AREA)

    # Convert to RGB for MediaPipe
    face_rgb = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(face_rgb)

    # Landmark detection and alignment
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            aligned_face = align_face(resized_face, face_landmarks.landmark)
            if aligned_face is not None:
                return prepare_face_tensor(aligned_face)

    return None


def draw_face_info(frame, track_id, identity, similarity, box):
    """
    Draw bounding box and identity label around a detected face.
    
    Args:
        frame (np.ndarray): Frame to draw on (BGR).
        track_id (int): Tracker ID assigned to the face.
        identity (str): Recognized identity name or "Stranger".
        similarity (float): Cosine similarity score with face bank.
        box (tuple): Bounding box coordinates (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = box

    color = (0, 0, 255) if identity == "Stranger" else (0, 255, 0)
    label = f"ID: {track_id} | Identity: {identity} | Sim: {similarity:.2f}"

    # Measure text width for centering
    (text_width, _), _ = cv2.getTextSize(
        label, 
        fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale = 0.8,
        thickness=2)


    # Center text above the bounding box
    center_x = (x1 + x2) // 2
    text_x = center_x - (text_width // 2)
    text_y = max(0, y1 - 10)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness = 2)

    cv2.putText(frame, 
                label, 
                (text_x, text_y), 
                fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = 0.8, 
                color = color, 
                thickness = 2, 
                lineType = cv2.LINE_AA)
    

def prepare_frame(frame, detector_result, args, face_mesh):
    """
    Prepare aligned face crops, track IDs, and bounding boxes from detection results.
    Args:
        frame (np.ndarray): Original video frame (BGR).
        detector_result: YOLO detector output for the current frame.
        args (Namespace): Parsed arguments containing thresholds and settings.
        face_mesh: Initialized MediaPipe FaceMesh for landmark detection.
    Returns:
        tuple: (aligned_faces, track_ids, bounding_boxes) if faces found, else None.
    """
    boxes = detector_result.boxes
    
    if not boxes:
        return None
    
    aligned_faces = []
    track_ids = []
    bounding_boxes = []
    
    for box in boxes:
        # Extract track ID (if available) and detection confidence
        track_id = int(box.id[0]) if box.id is not None else None
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        if conf < args.det_threshold:
            continue
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(detector_result.orig_shape[1], x2)
        y2 = min(detector_result.orig_shape[0], y2)
        
        # Crop face region and preprocess
        face_crop = frame[y1:y2, x1:x2]
        aligned = process_single_face(face_crop, face_mesh, min_face_size = args.min_face_size, blurry_threshold = args.blurry_threshold)

        if aligned is not None:
            aligned_faces.append(aligned)
            track_ids.append(track_id)
            bounding_boxes.append((x1, y1, x2, y2))

    if not aligned_faces:
        return None

    return aligned_faces, track_ids, bounding_boxes
            

def load_models(args, device):
    """
    Load the face detector, face recognizer, and facial landmark detector.
    Args:
        args: Parsed command-line arguments containing model paths and configs.
        device (torch.device): Target device to load the models on (CPU or CUDA).
    Returns:
        tuple:
            - detector (YOLO): Loaded YOLO face detector model.
            - recognizer (torch.nn.Module): Loaded face recognition model (EdgeFace).
            - face_mesh (mp.solutions.face_mesh.FaceMesh): MediaPipe FaceMesh landmark detector.
    """
    # Load YOLO face detector
    detector = YOLO(args.detector, verbose = False)

    # Load EdgeFace face recognizer model from GitHub
    recognizer = torch.hub.load(
        repo_or_dir = 'otroshi/edgeface',
        model = 'edgeface_s_gamma_05',
        source = 'github',
        pretrained = True
    ).to(device)
    recognizer.eval()

    # Initialize MediaPipe FaceMesh for facial landmarks
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode = False,
        refine_landmarks = True
    )

    return detector, recognizer, face_mesh

def parse_args():
    """
    Parse command-line arguments for real-time face detection and recognition.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    
    parser = argparse.ArgumentParser(description="Real-time Face Detection and Recognition")
    
    # Runtime settings
    parser.add_argument(
        "--mode", "-m",
        choices = ["amp", "no_amp"],
        default = "no_amp",
        help = "AMP: Automatic Mixed Precision ('amp') or standard mode ('no_amp'). Default is 'no_amp'."
    )
    parser.add_argument(
        "--source", "-s",
        type = str,
        default = "0",
        help = "Camera index (e.g., 0) or video file path. Default is '0' (webcam)."
    )

    # Model paths
    parser.add_argument(
        "--detector", "-d",
        type = str,
        default = "models/face-detector/YOLOv11/yolov11s-face.pt",
        help = "Path to YOLO face detector model weights."
    )
    parser.add_argument(
        "--tracker", "-t",
        type = str,
        default = "models/configs/bytetrack.yaml",
        help = "Path to tracker config file (default: ByteTrack YAML)."
    )
    parser.add_argument(
        "--face-bank", "-f",
        type = str,
        default = "data/face_banks/face-bank.pkl",
        help = "Path to face bank pickle file (default: 'data/face_banks/face-bank.pkl')."
    )

    # Detection and recognition thresholds
    parser.add_argument(
        "--det-threshold",
        type = float,
        default = 0.5,
        help = "Face detection confidence threshold (default: 0.5)."
    )
    parser.add_argument(
        "--rec-threshold",
        type = float,
        default = 0.5,
        help = "Face recognition similarity threshold (default: 0.5)."
    )
    parser.add_argument(
        "--blurry-threshold",
        type = float,
        default = 60.0,
        help = "Laplacian variance threshold for blur detection (default: 60.0). Lower values allow blurrier faces."
    )

    # Tracker settings
    parser.add_argument(
        "--min-track-age",
        type = int,
        default = 5,
        help = "Minimum number of frames before confirming an identity (default: 5)."
    )
    parser.add_argument(
        "--max-inactive-frames",
        type = int,
        default = 50,
        help = "Number of frames after which an inactive ID is removed (default: 50)."
    )

    # Face preprocessing settings
    parser.add_argument(
        "--min-face-size",
        type = int,
        default = 80,
        help = "Minimum face size (width and height in pixels) for recognition (default: 80)."
    )

    # Output and display settings
    parser.add_argument(
        "--show-fps",
        action = "store_true",
        help = "Overlay FPS counter on the video feed."
    )
    parser.add_argument(
        "--show-box",
        action = "store_true",
        help = "Draw bounding boxes even for unidentified faces."
    )
    parser.add_argument(
        "--save-video",
        action = "store_true",
        help = "Save the output video to disk instead of displaying only."
    )
    parser.add_argument(
        "--output-path",
        type = str,
        default = "output.mp4",
        help = "File path to save the output video (only used if --save-video is set)."
    )
    parser.add_argument(
        "--fps-smoothing",
        type = int,
        default = 30,
        help = "Number of frames over which FPS is averaged (default: 30)."
    )

    return parser.parse_args()


def save_video_writer(frame_shape, output_path):
    """
    Initialize a video writer to save output frames as a video file.
    Args:
        frame_shape (tuple): Shape of the video frames (height, width, channels).
        output_path (str): File path to save the output video (e.g., 'output.mp4').
    Returns:
        cv2.VideoWriter: Initialized video writer object.
    """
    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    fps = 30  
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def recognize_batch(aligned_faces, amp_enabled, recognizer, device):
    """
    Perform batched face recognition on aligned face tensors.
    Args:
        aligned_faces (List[torch.Tensor]): List of aligned face tensors (shape [3, 112, 112]).
        amp_enabled (bool): Whether to use Automatic Mixed Precision (AMP) for faster inference.
        recognizer (torch.nn.Module): Face recognition model.
        device (torch.device): Target device (CPU or CUDA).
    Returns:
        np.ndarray: Normalized face embeddings as a NumPy array (shape [batch_size, embedding_dim]).
    """
    batch = torch.stack(aligned_faces).to(device)

    # Use half precision only if AMP is enabled
    if amp_enabled:
        batch = batch.half()

    with torch.autocast(device_type = device.type, enabled = amp_enabled):
        embeddings = recognizer(batch)
        embeddings = F.normalize(embeddings, p = 2, dim = 1)  # L2-normalize embeddings
    embeddings = embeddings.detach().float().cpu().numpy()
    
    return embeddings

def main():
    """
    Main function to perform real-time face detection, tracking, and recognition.
    """
    args = parse_args()
    
    print(f"[INFO] torch.cuda.is_available() = {torch.cuda.is_available()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    amp_enabled = args.mode == "amp" and device.type == "cuda"
    if amp_enabled:
        print("[INFO] AMP is enabled.")
        
    # Load pre-built face embeddings
    # Initialize face detector, face recognizer, and facial landmark detector
    # Setup input source (camera index or video path).
    # Initialize identity tracker and FPS counter
    face_bank = load_face_bank(args.face_bank)
    detector, recognizer, face_mesh = load_models(args, device)
    source = int(args.source) if args.source.isdigit() else args.source
    results = detector.track(
        source = source,
        stream = True,
        persist = True,
        tracker = args.tracker,
        verbose = False)
    tracker = FaceTracker(recognize_threshold = args.rec_threshold)
    fps_counter = FPSCounter(smoothing = args.fps_smoothing)

    video_writer = None
    
    try:
        for result in results:
            fps_counter.update()
            tracker.increment_frame()
            
            frame = result.orig_img
            
            if video_writer is None and args.save_video:
                video_writer = save_video_writer(frame.shape, args.output_path)
            
            # Detect and align faces
            faces_info = prepare_frame(
                frame = frame, 
                detector_result = result, 
                args = args, 
                face_mesh = face_mesh)
            if not faces_info:
                continue
            
            aligned_faces, track_ids, bounding_boxes = faces_info
            
            if aligned_faces:
                # Generate embeddings
                embeddings = recognize_batch(aligned_faces, amp_enabled, recognizer, device)
                for emb, tid, (x1, y1, x2, y2) in zip(embeddings, track_ids, bounding_boxes):
                    identity, sim = match_face_embeddings(emb, face_bank, threshold = args.rec_threshold)
                    tracker.update(tid, identity, sim)
                    
                    # Confirm identity only after minimum tracking age
                    if tracker.get_age(tid) < args.min_track_age:
                        continue
                    
                    persistent_identity = tracker.get_identity(tid)
                    
                    if args.show_box or persistent_identity != "Stranger":
                        draw_face_info(
                            frame = frame,
                            track_id = tid,
                            identity = persistent_identity,
                            similarity = sim,
                            box =  (x1, y1, x2, y2))
                        
            if args.show_fps:
                fps_text = f"FPS: {fps_counter.get_fps():.1f}"
                cv2.putText(frame, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            
            if args.save_video and video_writer is not None:
                video_writer.write(frame)
                
            cv2.imshow("Face Detection + Recognition", frame)
            
            # Remove inactive tracks
            tracker.cleanup(max_inactive_frames = args.max_inactive_frames)
            
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        if args.save_video and video_writer is not None:
            video_writer.release()
        face_mesh.close()
        cv2.destroyAllWindows()
        sys.exit(0)
        
if __name__ == "__main__":
    main()