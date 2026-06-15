import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import difflib
import math

# --- CONFIGURATION ---
MODEL_PATH = 'best_model.pth'

# 1. Training Data (Required to rebuild the alphabet/vocabulary)
TRAIN_DATA_DIR = '../../02_Data_Processor/labeled_dataset_kaggle' 

# 2. Test Data (The new images you want to check)
TEST_DATA_DIR = '../../02_Data_Processor/labeled_dataset_test' 

IMG_HEIGHT = 32
IMG_WIDTH = 128

# Medicine Database for Spell Checking
MEDICINE_DB = [
    "Ace", "Aceta", "Alatrol", "Amodis", "Atrizin", "Axodin", "Az", "Azithrocin", "Azyth", 
    "Bacaid", "Backtone", "Baclofen", "Baclon", "Bacmax", "Beklo", "Bicozin", "Canazole", 
    "Candinil", "Cetisoft", "Conaz", "Dancel", "Denixil", "Diflu", "Dinafex", "Disopan", 
    "Esonix", "Esoral", "Etizin", "Exium", "Fenadin", "Fexo", "Fexofast", "Filmet", "Fixal",
    "Flamyd", "Flexibac", "Flexilax", "Flugal", "Ketocon", "Ketoral", "Ketotab", "Ketozol", 
    "Leptic", "Lucan-R", "Lumona", "M-Kast", "Maxima", "Maxpro", "Metro", "Metsina", "Monas", 
    "Montair", "Montene", "Montex", "Napa", "NapaExtend", "Nexcap", "Nexum", "Nidazyl", "Nizoder", 
    "Odmon", "Omastin", "Opton", "Progut", "Provair", "Renova", "Rhinil", "Ritch", "Rivotril", 
    "Romycin", "Rozith", "Sergel", "Tamen", "Telfast", "Tridosil", "Trilock", "Vifas", "Zithrin",
    "Algin", "Alphapress", "Arotml", "Artica", "B126", "Baemax", "Beltas", "Bilastin", "Bispro", 
    "Bukof", "Cefotil Plus", "Cinaron Plus", "Ciprin", "Clavurox", "Comet", "Cortimax", "D-Cap", 
    "Dermocin ointment", "Diapro MR", "Domilux", "Doxicap", "Doxiva", "Ebatin", "Ebion", "Edeloss", 
    "Erion Ointment", "Esonix", "Esonix M", "Esoral Mups", "Famodin", "Fenadin", "Filwel Gold", 
    "Filwel Teen HM", "Finix", "Flexilax", "Furoclav", "Gabarol-CR", "Gastrum", "Gaviflux DX", 
    "Hemofix FZ", "Indever", "Lingo", "Losarva", "Lubilax", "MAxsulin", "Maxcoral DX", "Maxpro", 
    "Maxpro Mups", "Menaril", "Mirapro", "Montair", "Montene", "Napa", "Napdas", "Olmezest", 
    "Ostocal", "Othera", "Oxat", "Perosa Cream", "Protinavit", "Radex", "Remmo", "Rex", "Rivotril", 
    "Rocipro", "Rocovay", "Rolac", "Sedil", "Sergel", "Telmidip", "Tenorix", "Trialon", "Tyclav", 
    "Veracal", "Visral", "XPA XR", "Xelpro Mups", "Xinc B", "Zolivox", "Zovia Silver", "ebatin", "traxcef"
]

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================

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
        # x shape: [Sequence_Length, Batch_Size, Embedding_Dim]
        return x + self.pe[:x.size(0), :].unsqueeze(1)

class OCRTransformer(nn.Module):
    def __init__(self, num_chars):
        super(OCRTransformer, self).__init__()
        
        # 1. CNN Backbone (Extracts visual features)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.MaxPool2d(2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.MaxPool2d(2, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.MaxPool2d((2, 1)), nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        # 2. Transformer Configuration
        self.d_model = 128
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=512, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. Projection (Bridge between CNN and Transformer)
        # Input features: 128 (channels) * 4 (height after pooling) = 512
        self.projection = nn.Linear(512, self.d_model)
        
        # 4. Final Classification
        self.linear = nn.Linear(self.d_model, num_chars + 1)
        
    def forward(self, x):
        features = self.cnn(x)
        b, c, h, w = features.size()
        
        # Rearrange for Transformer: [Width (Seq), Batch, Features]
        features = features.permute(3, 0, 1, 2).view(w, b, c * h)
        
        features = self.projection(features)
        features = self.pos_encoder(features)
        trans_out = self.transformer_encoder(features)
        
        # Output shape: [Sequence_Length, Batch, Num_Classes]
        return self.linear(trans_out)

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================

def get_vocab():
    """Scans the training folder to rebuild the character dictionary."""
    if not os.path.exists(TRAIN_DATA_DIR): 
        print(f"Error: Training directory not found at {TRAIN_DATA_DIR}")
        return 0, {}
        
    chars = set()
    files = [f for f in os.listdir(TRAIN_DATA_DIR) if f.lower().endswith(('.jpg', '.png'))]
    for f in files:
        label = f.split('_')[0]
        for c in label: chars.add(c)
    
    chars = sorted(list(chars))
    # CTC Loss needs 0 for blank, so we start map at 1
    return len(chars), {i+1: c for i, c in enumerate(chars)}

def decode(output, idx2char):
    """
    Decodes the model output into text.
    Expects output shape: [Sequence_Length, Batch_Size, Num_Classes]
    """
    arg_maxes = torch.argmax(output, dim=2) # Shape: [Seq, Batch]
    decodes = []
    
    # Iterate over batch items (usually just 1 item during inference)
    for i in range(arg_maxes.size(1)):
        seq = arg_maxes[:, i].tolist() # Get the full sequence for this batch item
        res = []
        prev = 0
        for idx in seq:
            # CTC Logic: Ignore blanks (0) and repeated characters
            if idx != 0 and idx != prev:
                res.append(idx2char[idx])
            prev = idx
        decodes.append(''.join(res))
        
    return decodes[0]

# ==========================================
# 3. MAIN EVALUATION LOOP
# ==========================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- TRANSFORMER EVALUATION ---")
    print(f"Device: {device}")

    # 1. Validation Checks
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return
    if not os.path.exists(TEST_DATA_DIR):
        print(f"Error: Test Data folder '{TEST_DATA_DIR}' not found.")
        return

    # 2. Build Vocabulary
    num_chars, idx2char = get_vocab()
    if num_chars == 0: return

    # 3. Load Model
    print(f"Loading model architecture...")
    model = OCRTransformer(num_chars).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR LOADING WEIGHTS: {e}")
        return

    # 4. Prepare Images
    files = [f for f in os.listdir(TEST_DATA_DIR) if f.lower().endswith(('.jpg', '.png'))]
    total = len(files)
    if total == 0:
        print(f"No images found in {TEST_DATA_DIR}")
        return

    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 5. Inference Loop
    count_pass = 0
    count_sys_pass = 0
    count_ai_pass = 0
    count_fail = 0

    print(f"\n{'FILENAME':<20} | {'TRANSFORMER':<15} | {'CORRECTED':<15} | {'RESULT'}")
    print("-" * 75)

    with torch.no_grad():
        for f in files:
            real_label = f.split('_')[0].split('.')[0]
            img_path = os.path.join(TEST_DATA_DIR, f)
            
            try:
                # Preprocess
                image = Image.open(img_path).convert('L')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Predict
                output = model(image_tensor)
                
                # --- CRITICAL FIX: REMOVED THE PERMUTE ---
                # Transformer output is already [Seq, Batch, Class], which decode() expects.
                
                raw_pred = decode(output, idx2char)
                
                # Spell Check
                matches = difflib.get_close_matches(raw_pred, MEDICINE_DB, n=1, cutoff=0.1)
                corrected = matches[0] if matches else "Unknown"
                
                # Scoring
                is_raw_correct = (raw_pred.lower() == real_label.lower())
                is_sys_correct = (corrected.lower() == real_label.lower())

                status = ""
                if is_raw_correct and is_sys_correct:
                    status = "PASS"
                    count_pass += 1
                elif is_raw_correct and not is_sys_correct:
                    status = "AI-PASS"
                    count_ai_pass += 1
                elif not is_raw_correct and is_sys_correct:
                    status = "SYS-PASS"
                    count_sys_pass += 1
                else:
                    status = "FAIL"
                    count_fail += 1

                print(f"{real_label[:19]:<20} | {raw_pred[:15]:<15} | {corrected[:15]:<15} | {status}")
                
            except Exception as e:
                print(f"{f[:19]:<20} | ERROR: {e}")

    # 6. Final Report
    print("\n" + "="*50)
    print("      TRANSFORMER MODEL PERFORMANCE")
    print("="*50)
    print(f"Total Test Images: {total}")
    print("-" * 50)
    
    raw_acc = ((count_pass + count_ai_pass) / total) * 100
    sys_acc = ((count_pass + count_sys_pass) / total) * 100
    
    print(f"1. Raw AI Accuracy:     {raw_acc:.2f}%")
    print(f"2. System Accuracy:     {sys_acc:.2f}%")
    print("-" * 50)
    print(f"Perfect: {count_pass} | System-Saved: {count_sys_pass} | AI-Good/Sys-Bad: {count_ai_pass} | Fail: {count_fail}")
    print("="*50)

if __name__ == "__main__":
    main()