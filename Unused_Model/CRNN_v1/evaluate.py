import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import difflib
from tqdm import tqdm  # Progress bar library

# --- CONFIG ---
MODEL_PATH = 'best_model.pth'
DATA_DIR = '../../02_Data_Processor/labeled_dataset'
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
"Romycin", "Rozith", "Sergel", "Tamen", "Telfast", "Tridosil", "Trilock", "Vifas", "Zithrin","Algin",
"Alphapress", "Arotml", "Artica", "B126", "Baemax", "Beltas", "Bilastin", "Bispro", "Bukof", "Cefotil Plus", 
"Cinaron Plus", "Ciprin", "Clavurox", "Comet", "Cortimax", "D-Cap", "Dermocin ointment", "Diapro MR", "Domilux", 
"Doxicap", "Doxiva", "Ebatin", "Ebion", "Edeloss", "Erion Ointment", "Esonix", "Esonix M", "Esoral Mups", "Famodin", 
"Fenadin", "Filwel Gold", "Filwel Teen HM", "Finix", "Flexilax", "Furoclav", "Gabarol-CR", "Gastrum", "Gaviflux DX", 
"Hemofix FZ", "Indever", "Lingo", "Losarva", "Lubilax", "MAxsulin", "Maxcoral DX", "Maxpro", "Maxpro Mups", "Menaril", 
"Mirapro", "Montair", "Montene", "Napa", "Napdas", "Olmezest", "Ostocal", "Othera", "Oxat", "Perosa Cream", "Protinavit", 
"Radex", "Remmo", "Rex", "Rivotril", "Rocipro", "Rocovay", "Rolac", "Sedil", "Sergel", "Telmidip", "Tenorix", "Trialon", 
"Tyclav", "Veracal", "Visral", "XPA XR", "Xelpro Mups", "Xinc B", "Zolivox", "Zovia Silver", "ebatin", "traxcef"
]
# --------------

# 1. MODEL CLASS (Must match training)
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
    if not os.path.exists(DATA_DIR): return 0, {}
    chars = set()
    for f in os.listdir(DATA_DIR):
        if f.endswith('.jpg'):
            # Filename format: Napa_uuid.jpg -> Label is "Napa"
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

# 3. MAIN EVALUATION LOOP
def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluator running on: {device}")
    
    # Load Model
    num_chars, idx2char = get_vocab()
    model = CRNN(num_chars).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Prepare Data
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')]
    total_images = len(files)
    
    if total_images == 0:
        print("No images found to test.")
        return

    raw_correct = 0
    system_correct = 0
    
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print(f"Testing on {total_images} images...")
    print("-" * 60)
    print(f"{'REAL LABEL':<15} | {'RAW AI PRED':<15} | {'CORRECTED':<15} | {'STATUS'}")
    print("-" * 60)

    # Loop through all files
    for f in files:
        # Get Real Label (Ground Truth)
        real_label = f.split('_')[0]
        
        # Load Image
        img_path = os.path.join(DATA_DIR, f)
        image = Image.open(img_path).convert('L')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # 1. Raw AI Prediction
        with torch.no_grad():
            output = model(image_tensor)
            output = output.permute(1, 0, 2)
            raw_pred = decode(output, idx2char)
            
        # 2. Apply Spell Checker
        matches = difflib.get_close_matches(raw_pred, MEDICINE_DB, n=1, cutoff=0.1)
        corrected_pred = matches[0] if matches else "Unknown"
        
        # 3. Check Accuracy
        # Check Raw
        if raw_pred.lower() == real_label.lower():
            raw_correct += 1
            
        # Check System (Corrected)
        is_system_correct = False
        if corrected_pred.lower() == real_label.lower():
            system_correct += 1
            is_system_correct = True
            
        # Print first 10 or errors for debugging
        # (Printing all might clutter terminal, but good to see errors)
        status = "OK" if is_system_correct else "FAIL"
        # Only print fails or first few to keep log clean
        # if not is_system_correct: 
        #    print(f"{real_label:<15} | {raw_pred:<15} | {corrected_pred:<15} | {status}")

    # --- FINAL REPORT ---
    raw_acc = (raw_correct / total_images) * 100
    sys_acc = (system_correct / total_images) * 100
    
    print("-" * 60)
    print("FINAL EVALUATION REPORT")
    print("-" * 60)
    print(f"Total Images Tested:    {total_images}")
    print(f"Raw AI Accuracy:        {raw_acc:.2f}%  (Without Spell Check)")
    print(f"System Accuracy:        {sys_acc:.2f}%  (With Spell Check)")
    print("-" * 60)
    
    if sys_acc < 70:
        print("Diagnosis: You need more training data or labeling.")
    elif sys_acc > 90:
        print("Diagnosis: Excellent! Ready for defense.")
    else:
        print("Diagnosis: Good prototype, passable for undergrad.")

if __name__ == "__main__":
    evaluate()