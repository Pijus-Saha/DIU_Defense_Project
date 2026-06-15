import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import math

# --- CONFIGURATION ---
# We point to the SAME dataset so the comparison is fair
DATA_DIR = '../../02_Data_Processor/labeled_dataset_kaggle' 
BATCH_SIZE = 32
EPOCHS = 300            
LEARNING_RATE = 0.001   # Transformers need lower learning rates
IMG_HEIGHT = 32
IMG_WIDTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ---------------------

print(f"Training Transformer Model on: {DEVICE}")

# 1. DATASET CLASS (Same as before)
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

# --- NEW: POSITIONAL ENCODING (Required for Transformers) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [SeqLen, Batch, D_Model]
        return x + self.pe[:x.size(0), :].unsqueeze(1)

# 2. DEFINE MODEL (CNN + TRANSFORMER)
class OCRTransformer(nn.Module):
    def __init__(self, num_chars):
        super(OCRTransformer, self).__init__()
        
        # Part 1: CNN Backbone (Same as CRNN for fair comparison)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.MaxPool2d(2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.MaxPool2d(2, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.MaxPool2d((2, 1)), nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        # Part 2: Feature Transformation
        # CNN output is 128 channels. We map this to d_model for Transformer
        self.d_model = 128
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)
        
        # Part 3: Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=512, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Part 4: Output Head
        self.linear = nn.Linear(self.d_model, num_chars + 1)
        
    def forward(self, x):
        # 1. Extract Features
        # x: [Batch, 1, 32, 128]
        features = self.cnn(x) 
        
        # 2. Reshape for Transformer
        # CNN Out: [Batch, Channels=128, Height=8, Width=32]
        b, c, h, w = features.size()
        
        # Permute to [Width, Batch, Channels*Height] -> We flatten Height into Channels if needed
        # But here, let's flatten Height and Channels together or just map
        features = features.permute(3, 0, 1, 2) # [Width, Batch, Channels, Height]
        features = features.view(w, b, c * h)   # [Width, Batch, Features]
        
        # Wait! CNN output height is 8. Channels is 128. 128*8 = 1024.
        # But our d_model is 128. We need a projection layer if sizes don't match.
        # Let's add a small projection to fix dimensions.
        if not hasattr(self, 'projection'):
            self.projection = nn.Linear(c * h, self.d_model).to(features.device)
            
        features = self.projection(features) # [Width, Batch, d_model]
        
        # 3. Add Positional Encoding
        features = self.pos_encoder(features)
        
        # 4. Pass through Transformer
        # Transformer expects [SeqLen, Batch, d_model]
        trans_out = self.transformer_encoder(features)
        
        # 5. Output
        output = self.linear(trans_out)
        
        # CTC Loss expects [SeqLen, Batch, Classes]
        return output

# 3. UTILITIES
def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images, 0)
    max_len = max([len(l) for l in labels])
    padded_labels = torch.full((len(batch), max_len), 0, dtype=torch.long)
    for i, l in enumerate(labels):
        padded_labels[i, :len(l)] = l
    return images, padded_labels, torch.tensor(label_lengths)

# 4. MAIN TRAINING LOOP
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
    
    model = OCRTransformer(len(char2idx)).to(DEVICE)
    criterion = nn.CTCLoss(blank=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_loss = float('inf')
    
    print("Starting Transformer Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for images, labels, label_lengths in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            preds = model(images) # [SeqLen, Batch, Classes]
            
            preds_log_softmax = preds.log_softmax(2)
            input_lengths = torch.full(size=(images.size(0),), fill_value=preds.size(0), dtype=torch.long)
            
            loss = criterion(preds_log_softmax, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, labels, label_lengths in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                preds = model(images)
                preds_log_softmax = preds.log_softmax(2)
                input_lengths = torch.full(size=(images.size(0),), fill_value=preds.size(0), dtype=torch.long)
                loss = criterion(preds_log_softmax, labels, input_lengths, label_lengths)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth') # Saves inside Transformer_v2 folder

    print("Transformer Training Complete.")

if __name__ == "__main__":
    main()