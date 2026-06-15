import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import difflib

# --- CONFIGURATION ---
MODEL_PATH = 'best_model.pth'

# 1. Where did we learn the alphabet? (Training Data)
TRAIN_DATA_DIR = '../../02_Data_Processor/labeled_dataset_kaggle' 

# 2. Where are the NEW images? (Untrained Data)
TEST_DATA_DIR = '../../02_Data_Processor/labeled_dataset_test' 

IMG_HEIGHT = 32
IMG_WIDTH = 128
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
# ---------------------

# 1. MODEL CLASS
class CRNN(nn.Module):
    def __init__(self, num_chars):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.MaxPool2d(2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.MaxPool2d(2, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1), nn.MaxPool2d((2, 1)), nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.rnn = nn.LSTM(512, 64, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(128, num_chars + 1)
        
    def forward(self, x):
        features = self.cnn(x)
        b, c, h, w = features.size()
        features = features.permute(0, 3, 1, 2).view(b, w, c * h)
        rnn_out, _ = self.rnn(features)
        return self.linear(rnn_out)

# 2. UTILS
def get_vocab():
    # We MUST check the TRAIN folder to know the alphabet
    if not os.path.exists(TRAIN_DATA_DIR):
        print(f"[Error] Training data not found at: {TRAIN_DATA_DIR}")
        return 0, {}
    
    chars = set()
    files = [f for f in os.listdir(TRAIN_DATA_DIR) if f.endswith('.jpg')]
    for f in files:
        label = f.split('_')[0]
        for c in label: chars.add(c)
    chars = sorted(list(chars))
    return len(chars), {i+1: c for i, c in enumerate(chars)}

def decode(output, idx2char):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    for i in range(arg_maxes.size(1)):
        seq = arg_maxes[:, i].tolist()
        res = []
        prev = 0
        for idx in seq:
            if idx != 0 and idx != prev:
                res.append(idx2char[idx])
            prev = idx
        decodes.append(''.join(res))
    return decodes[0]

# 3. PREDICTION LOOP
def predict_wild():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- UNTRAINED DATA TESTER ---")
    
    # Check folders
    if not os.path.exists(TEST_DATA_DIR):
        print(f"Error: Create this folder first: {TEST_DATA_DIR}")
        return

    # Load Model
    num_chars, idx2char = get_vocab()
    if num_chars == 0: return

    model = CRNN(num_chars).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get New Files
    files = [f for f in os.listdir(TEST_DATA_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(files) == 0:
        print(f"No images found in {TEST_DATA_DIR}")
        return

    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print(f"Found {len(files)} new images. Predicting...\n")
    print(f"{'FILENAME':<25} | {'RAW AI':<15} | {'CORRECTED':<15}")
    print("-" * 65)

    with torch.no_grad():
        for f in files:
            img_path = os.path.join(TEST_DATA_DIR, f)
            try:
                # Load & Process
                image = Image.open(img_path).convert('L')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Predict
                output = model(image_tensor)
                output = output.permute(1, 0, 2)
                raw_pred = decode(output, idx2char)
                
                # Spell Check
                matches = difflib.get_close_matches(raw_pred, MEDICINE_DB, n=1, cutoff=0.1)
                corrected_pred = matches[0] if matches else "Unknown"
                
                print(f"{f[:25]:<25} | {raw_pred:<15} | {corrected_pred:<15}")
                
            except Exception as e:
                print(f"{f[:25]:<25} | ERROR: {e}")

    print("-" * 65)

if __name__ == "__main__":
    predict_wild()