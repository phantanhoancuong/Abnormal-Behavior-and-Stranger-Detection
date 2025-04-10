import torch
import timm
import cv2
import os
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from sklearn.metrics import classification_report
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import numpy as np
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureExtractor(nn.Module):
    """
    Feature extraction using pre-trained EfficientNetV2-S model.
    
    Attributes:
        backbone (nn.Module): Pretrained EfficientNetV2-S model without classification head
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.backbone = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True, num_classes=0, global_pool='avg')

    def forward(self, x):
        """
        Forward pass for feature extraction.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Extracted features of shape (B, 1280)
        """
        return self.backbone(x)

        
class BiLSTMBlock(nn.Module):
    """
    Bidirectional LSTM block for temporal sequence processing.
    
    Attributes:
        bilstm (nn.LSTM): Bidirectional LSTM layer
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        """
        Args:
            input_size (int): Dimension of input features
            hidden_size (int): Number of hidden units
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
        """
        super(BiLSTMBlock, self).__init__()
        self.bilstm = nn.LSTM(input_size = input_size,
                              hidden_size = hidden_size,
                              num_layers = num_layers,
                              dropout = dropout,
                              bidirectional = True,
                              batch_first = True)
        
    def forward(self, x):
        """
        Forward pass through BiLSTM.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D_in)
            
        Returns:
            tuple: (output tensor, (hidden state, cell state))
        """
        output, (h_n, c_n) = self.bilstm(x)
        return output, (h_n, c_n)

class AttentionBlock(nn.Module):
    """
    Attention mechanism for temporal feature weighting.
    
    Attributes:
        attention_fc (nn.Linear): Linear layer for attention scoring
    """
    def __init__(self, hidden_size):
        """
        Args:
            hidden_size (int): Dimension of hidden states from BiLSTM
        """
        super(AttentionBlock, self).__init__()
        self.attention_fc = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, lstm_outputs):
        """
        Calculate attention-weighted context vector.
        
        Args:
            lstm_outputs (torch.Tensor): BiLSTM outputs of shape (B, T, 2*D_hid)
            
        Returns:
            tuple: (context vector, attention weights)
        """
        scores = self.attention_fc(lstm_outputs)
        attention_weights = F.softmax(scores, dim = 1)
        context_vector = torch.sum(attention_weights * lstm_outputs, dim = 1)
        return context_vector, attention_weights.squeeze(-1)
    
    
class ResidualBiLSTMBlock(nn.Module):
    """
    Residual BiLSTM block with attention mechanism.
    
    Attributes:
        bilstm_block (BiLSTMBlock): BiLSTM module
        attention_block (AttentionBlock): Attention module
        project (nn.Module): Linear projection for residual connection
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        """
        Args:
            input_size (int): Dimension of input features
            hidden_size (int): Number of hidden units in BiLSTM
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
        """
        super(ResidualBiLSTMBlock, self).__init__()
        
        self.bilstm_block = BiLSTMBlock(input_size, 
                                        hidden_size, 
                                        num_layers, 
                                        dropout)
        self.attention_block = AttentionBlock(hidden_size)
        
        if input_size != hidden_size * 2:
            self.project = nn.Linear(input_size, hidden_size * 2)
        else:
            self.project = nn.Identity()
            
    def forward(self, x):
        """
        Forward pass with residual connection.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D_in)
            
        Returns:
            tuple: (output tensor, attention weights)
        """
        bilstm_output, _ = self.bilstm_block(x)
        context_vector, attention_weights = self.attention_block(bilstm_output)
        
        residual = self.project(x.mean(dim = 1))
        output = context_vector + residual
        output = output.unsqueeze(1).repeat(1, x.shape[1], 1)  # (B, T, 2H)
        return output, attention_weights
        
class StackedResidualBiLSTM(nn.Module):
    """
    Stack of multiple ResidualBiLSTMBlocks for deep temporal modeling.
    
    Attributes:
        blocks (nn.ModuleList): List of ResidualBiLSTMBlocks
    """
    def __init__(self, num_blocks, input_size, hidden_size):
        """
        Args:
            num_blocks (int): Number of residual blocks to stack
            input_size (int): Input dimension for first block
            hidden_size (int): Hidden dimension for LSTM layers
        """
        super(StackedResidualBiLSTM, self).__init__()
        self.blocks = nn.ModuleList()
        
        for i in range(num_blocks):
            if i == 0:
                in_size = input_size
            else:
                in_size = hidden_size * 2
            self.blocks.append(ResidualBiLSTMBlock(input_size = in_size,
                                                    hidden_size = hidden_size,
                                                    num_layers = 2,
                                                    dropout = 0.2))
    
    def forward(self, x):
        """Sequentially process through all blocks"""
        for block in self.blocks:
            x, _ = block(x)
        return x
    
class ABDVideoDataset(Dataset):
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

class BehaviorClassificationModel(nn.Module):
    """
    Complete behavior classification model architecture.
    
    Attributes:
        feature_extractor (FeatureExtractor): CNN backbone
        temporal_model (StackedResidualBiLSTM): Temporal processor
        classifier (nn.Sequential): Final classification layers
    """
    def __init__(self, feature_dim, hidden_size, num_classes):
        """
        Args:
            feature_dim (int): Output dimension of feature extractor
            hidden_size (int): Hidden dimension for temporal model
            num_classes (int): Number of output classes
        """
        super(BehaviorClassificationModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.temporal_model = StackedResidualBiLSTM(num_blocks = 3,
                                                    input_size = feature_dim,
                                                    hidden_size = hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through full model.
        
        Args:
            x (torch.Tensor): Input video tensor (B, T, C, H, W)
            
        Returns:
            torch.Tensor: Class logits (B, num_classes)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)
        features = features.view(B, T, -1)
            
        temporal_output = self.temporal_model(features)
        video_repr = temporal_output.mean(dim = 1)
        output = self.classifier(video_repr)
        return output


def train_model(model, dataloader, criterion, optimizer, num_epochs, save_model_path):
    """
    Training loop for behavior classification model.
    
    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Model optimizer
        num_epochs (int): Number of training epochs
        save_model_path (str): Path to save trained weights
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for videos, labels in tqdm(dataloader, desc = f"Epoch {epoch + 1} / {num_epochs}"):
            videos = videos.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")
    
def evaluate_model(model, dataset, batch_size, frame_count, shuffle, save_eval_path):
    """
    Evaluation function for trained model.
    
    Args:
        model (nn.Module): Trained model
        dataset (Dataset): Evaluation dataset
        batch_size (int): Evaluation batch size
        frame_count (int): Frames per video
        shuffle (bool): Whether to shuffle data
        save_eval_path (str): Path to save evaluation report
    """
    model.eval()
    test_loader = DataLoader(dataset = dataset, 
                             batch_size = batch_size, 
                             shuffle = shuffle)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, labels in tqdm(test_loader, desc = "Evaluating"):
            padded_videos = []
            for v in videos:
                if v.shape[0] >= frame_count:
                    v = v[:frame_count]
                else:
                    pad_len = frame_count - v.shape[0]
                    padding = v[-1:].repeat(pad_len, 1, 1, 1)
                    v = torch.cat([v, padding], dim = 0)
                padded_videos.append(v)

            video_batch = torch.stack(padded_videos).to(torch.float32).to(device)
            labels_tensor = labels.clone().detach().to(device)

            outputs = model(video_batch)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_tensor.cpu().numpy())


    report = classification_report(
        all_labels,
        all_preds,
        target_names = dataset.classes,
        output_dict = True,
        zero_division = 0
    )
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names = dataset.classes, zero_division = 0))

    if save_eval_path:
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(save_eval_path, index = True)
        print(f"Results saved to {save_eval_path}")


def parse_args():
    parser = argparse.ArgumentParser(description = "Abnormal Behavior and Stranger Detection")
    parser.add_argument('--root_dir',
                        type = str,
                        required = True,
                        help = 'Path to dataset root folder')
    parser.add_argument('--frame_count', 
                        type = int,
                        default = 10,
                        help = 'Number of frames to extract per video (default: 10)')
    parser.add_argument('--batch_size',
                        type = int,
                        default = 1,
                        help = 'Batch size for DataLoader')
    parser.add_argument('--shuffle', 
                        type = lambda x: (str(x).lower() == 'true'), 
                        default = True, 
                        help = 'Shuffle the dataset (default: True)')
    parser.add_argument('--mode', 
                        type = str, 
                        choices = ['train', 'eval'], 
                        default = 'train', 
                        help = 'Mode: train or eval (default: train)')
    parser.add_argument('--num_epochs', 
                        type = int, 
                        default = 1, 
                        help = 'Number of training epochs (default: 1)')
    parser.add_argument('--learning_rate',
                        type = float,
                        default = 0.001,
                        help = 'Learning rate for training (default: 0.001)')
    parser.add_argument('--load_model_path', 
                        type = str, 
                        default = None, 
                        help = 'Path to the pre-trained model weights')
    parser.add_argument('--save_model_path', 
                        type = str,
                        default = 'trained_model.pth',
                        help = 'Path to save the trained model (default: model.pth)')
    parser.add_argument('--save_eval_path', 
                        type = str,
                        default = None,
                        help = 'Path to save CSV report')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate evaluation mode requirements
    if args.mode == 'eval':
            if not args.load_model_path or not os.path.isfile(args.load_model_path ):
                raise FileNotFoundError("--load_model_path provided is invalid.")
    
    # Initialize dataset and model
    dataset = ABDVideoDataset(root_dir = args.root_dir,
                              frame_count = args.frame_count)
    dataloader = DataLoader(dataset,
                            batch_size = args.batch_size,
                            shuffle = args.shuffle)
    num_classes = len(dataset.classes)
    model = BehaviorClassificationModel(feature_dim = 1280,
                                        hidden_size = 512,
                                        num_classes = num_classes)
    
    model.to(device)

    # Load pretrained weights if specified
    if args.load_model:
        print(f"Load model from {args.load_model_path}")
        model.load_state_dict(torch.load(args.load_model_path))
    
    # Execute requested mode
    if args.mode == 'train':
        print("Start training...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
        train_model(model, dataloader, criterion, optimizer, args.num_epochs, args.save_model_path)
        
    
    elif args.mode == 'eval':
        print("Start evaluating...")
        evaluate_model(model, dataset, batch_size = args.batch_size, frame_count = args.frame_count, shuffle = args.shuffle, save_eval_path = args.save_eval_path)
        
if __name__ == '__main__':
    main()