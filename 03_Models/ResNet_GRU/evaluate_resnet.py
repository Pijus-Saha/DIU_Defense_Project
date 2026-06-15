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

# FULL MEDICINE DATABASE (Paste the full list here)
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

# --- MODEL ARCHITECTURE ---
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes))
    def forward(self, x):
        return torch.relu(self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class ResNet_GRU(nn.Module):
    def __init__(self, num_chars):
        super(ResNet_GRU, self).__init__()
        self.in_planes = 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 2, stride=2)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=(2, 1))
        self.layer4 = self._make_layer(256, 2, stride=(2, 1))
        self.rnn = nn.GRU(512, 64, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(128, num_chars + 1)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        b, c, h, w = out.size()
        out = out.permute(0, 3, 1, 2).reshape(b, w, c * h)
        out, _ = self.rnn(out)
        return self.linear(out)

# --- UTILS ---
def get_vocab():
    if not os.path.exists(TRAIN_DATA_DIR): return 0, {}
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("--- RESNET-GRU EVALUATION ---")
    num_chars, idx2char = get_vocab()
    model = ResNet_GRU(num_chars).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error: {e}"); return

    files = [f for f in os.listdir(TEST_DATA_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    total = len(files)
    if total == 0: return

    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    count_pass = 0
    count_sys_pass = 0
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

                is_raw = (raw_pred.lower() == real_label.lower())
                is_sys = (corrected.lower() == real_label.lower())

                if is_raw and is_sys: count_pass += 1
                elif not is_raw and is_sys: count_sys_pass += 1
                else: count_fail += 1
            except: pass

    raw_acc = (count_pass / total) * 100
    sys_acc = ((count_pass + count_sys_pass) / total) * 100
    print(f"Raw AI Accuracy: {raw_acc:.2f}%")
    print(f"System Accuracy: {sys_acc:.2f}%")

if __name__ == "__main__":
    main()