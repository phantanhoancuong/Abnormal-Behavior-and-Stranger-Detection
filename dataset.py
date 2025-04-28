import os
import cv2
import torch
from torch.utils.data import Dataset

class BehaviorModelDataset(Dataset):
    """
    Dataset loader for abnormal behavior detection videos.
    
    Directory structure:
    root_dir/
        class1/
            videosA/
                videoA1.mpg
                videoA2.mpg
            videosB/
                videoB1.mpg
                videoB2.mpg
                videoB3.mpg
            ...
        class2/
            ...
    """
    def __init__(self, root_dir, frame_count):
        """
        Args:
            root_dir (str): Path to dataset directory
            frame_count (int): Number of frames per video sample
        """
        self.root_dir = root_dir
        self.transform = None
        self.classes = sorted(os.listdir(root_dir))
        self.frame_count = frame_count
        
        self.labels = []
        self.video_paths = []
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.idx_to_class = {idx: cls_name for idx, cls_name in enumerate(self.classes)}
    
        self.video_extensions = (".mpg", ".avi", ".mp4", ".mov", ".mkv")
        
        for class_name in self.classes:
            class_folder = os.path.join(root_dir, class_name)
            
            if not os.path.isdir(class_folder):
                continue
            
            for subfolder_name in os.listdir(class_folder):
                subfolder_path = os.path.join(class_folder, subfolder_name)
                
                if os.path.isdir(subfolder_path):
                    has_video = any(
                        file_name.lower().endswith(self.video_extensions)
                        for file_name in os.listdir(subfolder_path)
                    )
                if not has_video:
                    continue
                
                for file_name in os.listdir(subfolder_path):
                    if file_name.lower().endswith(self.video_extensions):
                        video_path = os.path.join(subfolder_path, file_name)
                        self.video_paths.append(video_path)
                        self.labels.append(self.class_to_idx[class_name])
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        Load and process video.
        
        Returns:
            tuple: (video tensor, label)
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (384, 384))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.tensor(frame).permute(2, 0, 1)
            frame = frame.float() / 255.0
            frames.append(frame)
        
        cap.release()
        
        if len(frames) >= self.frame_count:
            frames = frames[:self.frame_count]
        else:
            pad_len = self.frame_count - len(frames)
            padding = [frames[-1]] * pad_len if frames else [torch.zeros(3, 384, 384)] * self.frame_count
            frames.extend(padding)
        
        
        video_tensor = torch.stack(frames)
        
        return video_tensor, label
