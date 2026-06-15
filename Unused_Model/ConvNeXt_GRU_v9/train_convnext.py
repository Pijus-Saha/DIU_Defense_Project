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
BATCH_SIZE = 16  # ConvNeXt is slightly heavy, safe batch size
EPOCHS = 100
LEARNING_RATE = 0.0001
IMG_HEIGHT = 32
IMG_WIDTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --------------

print(f"Training ConvNeXt-Tiny-GRU on: {DEVICE}")

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

# 2. MODEL: ConvNeXt-Tiny + GRU
class ConvNeXt_GRU(nn.Module):
    def __init__(self, num_chars):
        super(ConvNeXt_GRU, self).__init__()
        
        # Load ConvNeXt Tiny (Weights=None to train from scratch for custom dims)
        self.convnext = models.convnext_tiny(weights=None)
        
        # 1. Modify STEM (Input Layer)
        # Original: Conv2d(3, 96, 4, stride=4) -> Shrinks by 4x immediately
        # Our Fix: Conv2d(1, 96, 3, stride=1) -> Keep resolution high initially
        self.convnext.features[0][0] = nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=1)
        
        # 2. MODIFY DOWNSAMPLING LAYERS (To preserve Width)
        # ConvNeXt features structure:
        # [0]: Stem
        # [1]: Stage 1
        # [2]: Downsample 1 (Normal stride 2)
        # [3]: Stage 2
        # [4]: Downsample 2 (Normal stride 2) -> CHANGE TO (2, 1)
        # [5]: Stage 3
        # [6]: Downsample 3 (Normal stride 2) -> CHANGE TO (2, 1)
        # [7]: Stage 4
        
        # Fix Downsample 2 (Index 4)
        # Original: LayerNorm -> Conv2d(96, 192, 2, stride=2)
        self.convnext.features[4][1] = nn.Conv2d(192, 384, kernel_size=2, stride=(2, 1))
        
        # Fix Downsample 3 (Index 6)
        # Original: LayerNorm -> Conv2d(384, 768, 2, stride=2)
        self.convnext.features[6][1] = nn.Conv2d(384, 768, kernel_size=2, stride=(2, 1))
        
        # Final Output Calculation:
        # Input: 32x128
        # Stem (s1): 32x128
        # Stage 1: 32x128
        # Downsample 1 (s2): 16x64
        # Stage 2: 16x64
        # Downsample 2 (s2,1): 8x64
        # Stage 3: 8x64
        # Downsample 3 (s2,1): 4x64
        # Stage 4: 4x64
        
        # Final Channels: 768
        # Final Height: 4
        # Total Features: 768 * 4 = 3072
        
        self.adapter = nn.Linear(3072, 512)
        self.rnn = nn.GRU(512, 64, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(128, num_chars + 1)

    def forward(self, x):
        features = self.convnext.features(x)
        
        b, c, h, w = features.size()
        # Reshape
        features = features.permute(0, 3, 1, 2).reshape(b, w, c * h)
        
        features = self.adapter(features)
        out, _ = self.rnn(features)
        return self.linear(out)

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
    
    model = ConvNeXt_GRU(len(char2idx)).to(DEVICE)
    criterion = nn.CTCLoss(blank=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_loss = float('inf')
    
    print("Starting ConvNeXt-GRU Training...")
    
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
            
            # Safety check
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