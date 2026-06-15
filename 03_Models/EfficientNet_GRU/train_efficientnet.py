import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

# --- CONFIG ---
DATA_DIR = '../../02_Data_Processor/labeled_dataset_kaggle' 
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0005 
IMG_HEIGHT = 32
IMG_WIDTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --------------

print(f"Training EfficientNet-B0-GRU (Fixed Dims) on: {DEVICE}")

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

# 2. MODEL: EfficientNet-B0 + GRU (With Stride Hacks)
class EfficientNet_GRU(nn.Module):
    def __init__(self, num_chars):
        super(EfficientNet_GRU, self).__init__()
        
        # Load Weights
        self.effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # 1. Modify Stem (First Layer) for Grayscale
        # Original stride was 2. We change to (2, 1) to keep width.
        self.effnet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        
        # 2. STRIDE HACKING
        # Force strides to (2, 1) or (1, 1) to preserve Width.
        self.effnet.features[2][0].block[1][0].stride = (2, 1)
        self.effnet.features[3][0].block[1][0].stride = (2, 1)
        self.effnet.features[4][0].block[1][0].stride = (2, 1)
        self.effnet.features[6][0].block[1][0].stride = (1, 1)
        
        # 3. Adapter Layer
        # Output Channels: 1280
        # Output Height: 2
        # Total Features: 1280 * 2 = 2560
        self.adapter = nn.Linear(2560, 512) # <--- FIXED THIS NUMBER
        
        self.rnn = nn.GRU(512, 64, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(128, num_chars + 1)

    def forward(self, x):
        features = self.effnet.features(x)
        
        b, c, h, w = features.size()
        
        # Reshape [Batch, Width, Channels*Height]
        features = features.permute(0, 3, 1, 2).reshape(b, w, c * h)
        
        features = self.adapter(features)
        
        out, _ = self.rnn(features)
        out = self.linear(out)
        return out

# 3. UTILS
def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images, 0)
    max_len = max([len(l) for l in labels])
    padded_labels = torch.full((len(batch), max_len), 0, dtype=torch.long)
    for i, l in enumerate(labels):
        padded_labels[i, :len(l)] = l
    return images, padded_labels, torch.tensor(label_lengths)

# 4. TRAINING
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
    
    model = EfficientNet_GRU(len(char2idx)).to(DEVICE)
    criterion = nn.CTCLoss(blank=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_loss = float('inf')
    
    print("Starting Fixed EfficientNet Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for images, labels, label_lengths in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            preds = model(images)
            preds = preds.permute(1, 0, 2)
            preds_log_softmax = preds.log_softmax(2)
            
            input_lengths = torch.full(size=(images.size(0),), fill_value=preds.size(0), dtype=torch.long)
            
            valid_indices = []
            for i in range(len(labels)):
                if label_lengths[i] <= preds.size(0): valid_indices.append(i)
            if len(valid_indices) < len(labels): continue

            loss = criterion(preds_log_softmax, labels, input_lengths, label_lengths)
            loss.backward()
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
            if not np.isnan(val_loss):
                torch.save(model.state_dict(), 'best_model.pth')

    print("Training Complete.")

if __name__ == "__main__":
    main()