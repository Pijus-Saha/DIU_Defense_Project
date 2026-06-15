import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import difflib

# --- CONFIGURATION ---
MODEL_PATH = 'best_model.pth'
TRAIN_DATA_DIR = '../../02_Data_Processor/labeled_dataset_kaggle' 
TEST_DATA_DIR = '../../02_Data_Processor/labeled_dataset_test' 

IMG_HEIGHT = 32
IMG_WIDTH = 128

# FULL MEDICINE DATABASE
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

# --- MODEL ARCHITECTURE (Must match training exactly) ---
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
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.features = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 128, stride=1),
            nn.MaxPool2d((2, 1)),
            DepthwiseSeparableConv(128, 256, stride=1),
            nn.MaxPool2d((2, 1)),
            DepthwiseSeparableConv(256, 256, stride=1)
        )
        
        self.rnn = nn.GRU(512, 64, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(128, num_chars + 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.features(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.linear(x)
        return x

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

# --- MAIN EVALUATION LOOP ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- MOBILENET-GRU EVALUATION ---")
    
    if not os.path.exists(TEST_DATA_DIR):
        print(f"[Error] Test folder not found: {TEST_DATA_DIR}")
        return

    num_chars, idx2char = get_vocab()
    if num_chars == 0:
        print("[Error] Could not build vocabulary.")
        return

    model = MobileNet_GRU(num_chars).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except Exception as e:
        print(f"[Error] Model loading failed: {e}")
        return

    files = [f for f in os.listdir(TEST_DATA_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    total = len(files)
    
    print(f"Testing {total} images...")
    
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    count_pass = 0
    count_sys_pass = 0
    count_ai_pass = 0
    count_fail = 0

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
                
            except: pass

    # --- FINAL REPORT ---
    print("\n" + "="*50)
    print("      MOBILENET MODEL PERFORMANCE")
    print("="*50)
    print(f"Total Test Images: {total}")
    print("-" * 50)
    
    raw_acc = ((count_pass + count_ai_pass) / total) * 100
    sys_acc = ((count_pass + count_sys_pass) / total) * 100
    
    print(f"1. Raw AI Accuracy:     {raw_acc:.2f}%")
    print(f"2. System Accuracy:     {sys_acc:.2f}%")
    print("-" * 50)
    print(f"Pass: {count_pass} | Sys-Pass: {count_sys_pass} | AI-Pass: {count_ai_pass} | Fail: {count_fail}")
    print("="*50)

if __name__ == "__main__":
    main()