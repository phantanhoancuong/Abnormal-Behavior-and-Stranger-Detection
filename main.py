import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse

from dataset import BehaviorModelDataset
from behavior_model import BehaviorModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    


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
    dataset = BehaviorModelDataset(root_dir = args.root_dir,
                                   frame_count = args.frame_count)
    dataloader = DataLoader(dataset,
                            batch_size = args.batch_size,
                            shuffle = args.shuffle)
    num_classes = len(dataset.classes)
    model = BehaviorModel(feature_dim = 1280,
                          hidden_size = 512,
                          num_classes = num_classes)
    
    model.to(device)

    # Load pretrained weights if specified
    if args.load_model_path:
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