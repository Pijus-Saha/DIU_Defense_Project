import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

# --- CONFIG ---
# Correct path with underscore
DATA_DIR = '../../02_Data_Processor/labeled_dataset_kaggle' 
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001 # We can use higher LR now because we are training from scratch
IMG_HEIGHT = 32
IMG_WIDTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --------------

print(f"Training Custom MobileNet-GRU on: {DEVICE}")

# 1. DATASET
class PrescriptionDataset(Dataset):
    def __init__(self, image_paths, labels, char2idx, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.char2idx = char2idx
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('L') 
            if self.transform: image = self.transform(image)
            label = self.labels[idx]
            label_seq = [self.char2idx[char] for char in label]
            return image, torch.tensor(label_seq, dtype=torch.long), len(label_seq)
        except:
            return self.__getitem__((idx + 1) % len(self))

def load_data(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    paths = []
    labels = []
    chars = set()
    for f in files:
        label = f.split('_')[0]
        paths.append(os.path.join(directory, f))
        labels.append(label)
        for c in label: chars.add(c)
    chars = sorted(list(chars))
    char2idx = {c: i+1 for i, c in enumerate(chars)}
    idx2char = {i+1: c for i, c in enumerate(chars)}
    return paths, labels, char2idx, idx2char

# 2. MODEL: Custom MobileNet-Style Architecture
# This uses "Depthwise Separable Convolutions" (The secret sauce of MobileNet)
# But we control the stride so it doesn't shrink the image too much.

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MobileNet_GRU(nn.Module):
    def __init__(self, num_chars):
        super(MobileNet_GRU, self).__init__()
        
        # Initial Conv
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # MobileNet Blocks (Depthwise Separable)
        # We keep strides conservative to maintain width > 16
        self.features = nn.Sequential(
            # Stride 2 (32x128 -> 16x64)
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 64, stride=1),
            
            # Stride 2 (16x64 -> 8x32)
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 128, stride=1),
            
            # Pool Height Only (8x32 -> 4x32)
            nn.MaxPool2d((2, 1)),
            DepthwiseSeparableConv(128, 256, stride=1),
            
            # Pool Height Only (4x32 -> 2x32)
            nn.MaxPool2d((2, 1)),
            DepthwiseSeparableConv(256, 256, stride=1)
        )
        # Final Output: [Batch, 256, 2, 32]
        # Width is 32. Plenty of space for "Azithromycin".
        
        # RNN
        # Input dim: 256 channels * 2 height = 512
        self.rnn = nn.GRU(512, 64, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(128, num_chars + 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.features(x)
        
        b, c, h, w = x.size()
        # Reshape for RNN [Batch, Width, Channels*Height]
        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        
        x, _ = self.rnn(x)
        x = self.linear(x)
        return x

# 3. UTILS
def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images, 0)
    max_len = max([len(l) for l in labels])
    padded_labels = torch.full((len(batch), max_len), 0, dtype=torch.long)
    for i, l in enumerate(labels):
        padded_labels[i, :len(l)] = l
    return images, padded_labels, torch.tensor(label_lengths)

# 4. TRAINING LOOP
def main():
    all_paths, all_labels, char2idx, idx2char = load_data(DATA_DIR)
    train_paths, val_paths, train_labels, val_labels = train_test_split(all_paths, all_labels, test_size=0.1)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomRotation(5, fill=255),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_loader = DataLoader(PrescriptionDataset(train_paths, train_labels, char2idx, transform), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(PrescriptionDataset(val_paths, val_labels, char2idx, transform), batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    model = MobileNet_GRU(len(char2idx)).to(DEVICE)
    criterion = nn.CTCLoss(blank=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_loss = float('inf')
    print("Starting Custom MobileNet-GRU Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for images, labels, label_lengths in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            preds = model(images)
            # [Time, Batch, Classes]
            preds = preds.permute(1, 0, 2) 
            preds_log_softmax = preds.log_softmax(2)
            
            input_lengths = torch.full(size=(images.size(0),), fill_value=preds.size(0), dtype=torch.long)
            
            # FIX: Ensure target length <= input length (Safety check)
            # This prevents NaN if a word is exceptionally long
            valid_indices = []
            for i in range(len(labels)):
                if label_lengths[i] <= preds.size(0):
                    valid_indices.append(i)
            
            if len(valid_indices) < len(labels):
                # Skip batch if data is bad (rare safety valve)
                continue 

            loss = criterion(preds_log_softmax, labels, input_lengths, label_lengths)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, labels, label_lengths in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                preds = model(images)
                preds = preds.permute(1, 0, 2)
                preds_log_softmax = preds.log_softmax(2)
                input_lengths = torch.full(size=(images.size(0),), fill_value=preds.size(0), dtype=torch.long)
                loss = criterion(preds_log_softmax, labels, input_lengths, label_lengths)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            # Avoid saving NaN
            if not np.isnan(val_loss):
                torch.save(model.state_dict(), 'best_model.pth')

    print("Training Complete.")

if __name__ == "__main__":
    main()