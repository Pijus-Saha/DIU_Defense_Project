import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import difflib

# --- CONFIGURATION ---
MODEL_PATH = 'best_model.pth'
# Ensure this matches your folder structure exactly
TRAIN_DATA_DIR = '../../02_Data_Processor/labeled_dataset_kaggle' 
TEST_DATA_DIR = '../../02_Data_Processor/labeled_dataset_test' 

IMG_HEIGHT = 32
IMG_WIDTH = 128

# Paste your FULL medicine list here
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

# --- MODEL: GRU Architecture ---
class CRNN_GRU(nn.Module):
    def __init__(self, num_chars):
        super(CRNN_GRU, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.MaxPool2d(2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.MaxPool2d(2, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.MaxPool2d((2, 1)), nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.rnn = nn.GRU(512, 64, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(128, num_chars + 1)
        
    def forward(self, x):
        features = self.cnn(x)
        b, c, h, w = features.size()
        features = features.permute(0, 3, 1, 2).view(b, w, c * h)
        rnn_out, _ = self.rnn(features)
        return self.linear(rnn_out)

# --- UTILS ---
def get_vocab():
    if not os.path.exists(TRAIN_DATA_DIR): return 0, {}
    chars = set()
    files = [f for f in os.listdir(TRAIN_DATA_DIR) if f.lower().endswith(('.jpg', '.png'))]
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- DETAILED GRU EVALUATION ---")
    
    # 1. Check Folder
    if not os.path.exists(TEST_DATA_DIR):
        print(f"[Error] Test folder not found: {os.path.abspath(TEST_DATA_DIR)}")
        return

    num_chars, idx2char = get_vocab()
    if num_chars == 0:
        print("[Error] Could not build vocabulary from Train folder.")
        return

    # 2. Load Model
    model = CRNN_GRU(num_chars).to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] best_model.pth not found in GRU_v3 folder.")
        return

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except Exception as e:
        print(f"[Error] Model load failed: {e}")
        return

    # 3. Find Images (Robust Search)
    files = [f for f in os.listdir(TEST_DATA_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    total = len(files)
    
    if total == 0:
        print(f"[Error] No images found in: {TEST_DATA_DIR}")
        return
    
    print(f"Testing {total} images...")

    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Counters
    count_pass = 0
    count_sys_pass = 0
    count_ai_pass = 0
    count_fail = 0

    print(f"\n{'FILENAME':<20} | {'RAW GRU':<15} | {'CORRECTED':<15} | {'RESULT'}")
    print("-" * 70)

    with torch.no_grad():
        for f in files:
            real_label = f.split('_')[0].split('.')[0]
            img_path = os.path.join(TEST_DATA_DIR, f)
            try:
                image = Image.open(img_path).convert('L')
                image = transform(image).unsqueeze(0).to(device)
                
                output = model(image)
                output = output.permute(1, 0, 2)
                raw_pred = decode(output, idx2char)
                
                matches = difflib.get_close_matches(raw_pred, MEDICINE_DB, n=1, cutoff=0.1)
                corrected = matches[0] if matches else "Unknown"

                # Logic Breakdown
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

                print(f"{real_label:<20} | {raw_pred:<15} | {corrected:<15} | {status}")
                
            except Exception as e:
                print(f"{f:<20} | Error: {e}")

    # --- FINAL REPORT ---
    print("\n" + "="*50)
    print("      GRU MODEL PERFORMANCE REPORT")
    print("="*50)
    print(f"Total Test Images: {total}")
    print("-" * 50)
    
    raw_acc = ((count_pass + count_ai_pass) / total) * 100
    sys_acc = ((count_pass + count_sys_pass) / total) * 100
    
    print(f"1. Raw GRU Accuracy:    {raw_acc:.2f}%")
    print(f"2. System Accuracy:     {sys_acc:.2f}%")
    print("-" * 50)
    print(f"Pass: {count_pass} | Sys-Pass: {count_sys_pass} | AI-Pass: {count_ai_pass} | Fail: {count_fail}")
    print("="*50)

if __name__ == "__main__":
    main()