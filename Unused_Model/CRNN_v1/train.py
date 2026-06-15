import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

# --- CONFIGURATION ---
DATA_DIR = '../../02_Data_Processor/labeled_dataset'  # Folder with Napa_uuid.jpg
BATCH_SIZE = 16
EPOCHS = 200                    # Increase this to 100+ later
LEARNING_RATE = 0.001
IMG_HEIGHT = 32
IMG_WIDTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ---------------------

print(f"Running on: {DEVICE}")

# 1. PREPARE DATASET
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
        image = Image.open(img_path).convert('L') # Grayscale
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        # Convert label to integer sequence
        label_seq = [self.char2idx[char] for char in label]
        
        return image, torch.tensor(label_seq, dtype=torch.long), len(label_seq)

# Helper to load files and parse "Napa_uuid.jpg" -> "Napa"
def load_data(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    paths = []
    labels = []
    chars = set()
    
    for f in files:
        # File name format: Word_UUID.jpg
        # We split by '_' and take the first part
        label = f.split('_')[0]
        paths.append(os.path.join(directory, f))
        labels.append(label)
        for c in label:
            chars.add(c)
            
    # Create vocabulary
    chars = sorted(list(chars))
    char2idx = {c: i+1 for i, c in enumerate(chars)} # 0 is reserved for CTC blank
    idx2char = {i+1: c for i, c in enumerate(chars)}
    
    print(f"Loaded {len(files)} images.")
    print(f"Vocabulary ({len(chars)} chars): {''.join(chars)}")
    
    return paths, labels, char2idx, idx2char

# 2. DEFINE MODEL (CRNN)
# 2. DEFINE MODEL (CRNN) - CORRECTED
class CRNN(nn.Module):
    def __init__(self, num_chars):
        super(CRNN, self).__init__()
        
        # CNN Part (Feature Extraction)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2), 
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 1)), # Vertical pooling only
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        # RNN Part (Sequence Modeling)
        # --- FIX IS HERE: Changed 128 to 512 ---
        self.rnn = nn.LSTM(512, 64, bidirectional=True, batch_first=True)
        
        # Output Part
        self.linear = nn.Linear(128, num_chars + 1) # +1 for CTC Blank
        
    def forward(self, x):
        # x shape: [Batch, 1, 32, 128]
        features = self.cnn(x) 
        
        # Reshape for RNN: [Batch, TimeStep, Features]
        b, c, h, w = features.size()
        features = features.permute(0, 3, 1, 2) # [Batch, w, c, h]
        features = features.view(b, w, c * h)   # Flatten: 128 * 4 = 512
        
        # Pass to RNN
        rnn_out, _ = self.rnn(features)
        
        # Linear layer
        output = self.linear(rnn_out)
        return output
# 3. UTILITIES
def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images, 0)
    
    # Pad labels to same length for batch processing
    max_len = max([len(l) for l in labels])
    padded_labels = torch.full((len(batch), max_len), 0, dtype=torch.long)
    
    for i, l in enumerate(labels):
        padded_labels[i, :len(l)] = l
        
    return images, padded_labels, torch.tensor(label_lengths)

def decode_prediction(output, idx2char):
    # Greedy Decoder for CTC
    # output shape: [Time, Batch, Classes]
    arg_maxes = torch.argmax(output, dim=2) # [Time, Batch]
    decodes = []
    
    for i in range(arg_maxes.size(1)): # For each item in batch
        seq = arg_maxes[:, i].tolist()
        result = []
        prev = 0
        for idx in seq:
            if idx != 0 and idx != prev: # 0 is blank, ignore duplicates
                result.append(idx2char[idx])
            prev = idx
        decodes.append(''.join(result))
    return decodes

# 4. MAIN TRAINING LOOP
def main():
    # Load Data
    all_paths, all_labels, char2idx, idx2char = load_data(DATA_DIR)
    
    # Split Data (80% Train, 20% Test)
    train_paths, val_paths, train_labels, val_labels = train_test_split(all_paths, all_labels, test_size=0.2, random_state=42)
    
    # NEW: Stronger Augmentation for small datasets
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        # Randomly rotate the image slightly (-5 to +5 degrees)
        transforms.RandomRotation(5, fill=255), 
        # Randomly change brightness/contrast to simulate bad lighting
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = PrescriptionDataset(train_paths, train_labels, char2idx, transform)
    val_dataset = PrescriptionDataset(val_paths, val_labels, char2idx, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Initialize Model
    model = CRNN(len(char2idx)).to(DEVICE)
    criterion = nn.CTCLoss(blank=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting Training...")
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for images, labels, label_lengths in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            preds = model(images) # [Batch, Time, Classes]
            preds = preds.permute(1, 0, 2) # CTCLoss expects [Time, Batch, Classes]
            preds_log_softmax = preds.log_softmax(2)
            
            # Calculate Loss
            input_lengths = torch.full(size=(images.size(0),), fill_value=preds.size(0), dtype=torch.long)
            loss = criterion(preds_log_softmax, labels, input_lengths, label_lengths)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for i, (images, labels, label_lengths) in enumerate(val_loader):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                preds = model(images)
                preds = preds.permute(1, 0, 2)
                preds_log_softmax = preds.log_softmax(2)
                input_lengths = torch.full(size=(images.size(0),), fill_value=preds.size(0), dtype=torch.long)
                loss = criterion(preds_log_softmax, labels, input_lengths, label_lengths)
                val_loss += loss.item()

                # --- NEW DEBUG PRINT ---
                # Only print for the first batch of every epoch
                if i == 0:
                    texts = decode_prediction(preds, idx2char)
                    print(f"   AI sees: '{texts[0]}'  (Real Label length: {label_lengths[0]})")
                # -----------------------

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    print("Training Complete. Model saved as 'best_model.pth'")

if __name__ == "__main__":
    main()