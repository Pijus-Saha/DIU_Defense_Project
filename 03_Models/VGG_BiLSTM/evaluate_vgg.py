import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import difflib

# --- CONFIG ---
MODEL_PATH = 'best_model.pth'
TRAIN_DATA_DIR = '../../02_Data_Processor/labeled_dataset_kaggle' 
TEST_DATA_DIR = '../../02_Data_Processor/labeled_dataset_test' 
IMG_HEIGHT = 32
IMG_WIDTH = 128

# --- FULL MEDICINE DATABASE (The missing piece!) ---
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

# --- MODEL ARCHITECTURE (VGG + BiLSTM) ---
class VGG_BiLSTM(nn.Module):
    def __init__(self, num_chars):
        super(VGG_BiLSTM, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True), nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True), nn.BatchNorm2d(512), nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.BatchNorm2d(512), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(True)
        )
        self.rnn = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(512, num_chars + 1)

    def forward(self, x):
        conv = self.features(x)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv)
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

# --- MAIN ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- VGG MODEL EVALUATION ---")
    
    num_chars, idx2char = get_vocab()
    model = VGG_BiLSTM(num_chars).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error: {e}")
        return

    files = [f for f in os.listdir(TEST_DATA_DIR) if f.lower().endswith(('.jpg', '.png'))]
    total = len(files)
    
    if total == 0:
        print("No images found!")
        return

    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    count_sys_pass = 0
    count_pass = 0
    
    with torch.no_grad():
        for f in files:
            try:
                image = Image.open(os.path.join(TEST_DATA_DIR, f)).convert('L')
                image = transform(image).unsqueeze(0).to(device)
                
                output = model(image)
                output = output.permute(1, 0, 2) # Ensure CTC dims
                raw_pred = decode(output, idx2char)
                
                matches = difflib.get_close_matches(raw_pred, MEDICINE_DB, n=1, cutoff=0.1)
                corrected = matches[0] if matches else "Unknown"
                
                real_label = f.split('_')[0].split('.')[0]
                
                if corrected.lower() == real_label.lower(): count_sys_pass += 1
                if raw_pred.lower() == real_label.lower(): count_pass += 1
            except: pass

    print(f"Raw Accuracy: {(count_pass/total)*100:.2f}%")
    print(f"System Accuracy: {(count_sys_pass/total)*100:.2f}%")

if __name__ == "__main__":
    main()