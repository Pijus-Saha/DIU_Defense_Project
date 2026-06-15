import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import difflib
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import pandas as pd

# --- CONFIGURATION ---
TRAIN_DIR = '../02_Data_Processor/labeled_dataset_kaggle' 
TEST_DIR = '../02_Data_Processor/labeled_dataset_test' 
NLP_MODEL_PATH = 'NLP_Corrector/nlp_corrector.pth' 
MODELS_ROOT_DIR = '.' # Where to scan for model folders

IMG_HEIGHT = 32
IMG_WIDTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# =============================================================================
# 1. VISION MODEL ARCHITECTURES (ALL 8)
# =============================================================================

# --- A. CRNN (LSTM) ---
class CRNN(nn.Module):
    def __init__(self, num_chars):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1), nn.MaxPool2d(2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 1, 1), nn.MaxPool2d(2, 2), nn.ReLU(), nn.Conv2d(64, 128, 3, 1, 1), nn.MaxPool2d((2, 1)), nn.ReLU(), nn.BatchNorm2d(128))
        self.rnn = nn.LSTM(512, 64, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(128, num_chars + 1)
    def forward(self, x):
        features = self.cnn(x); features = features.permute(0, 3, 1, 2).view(features.size(0), features.size(3), -1)
        rnn_out, _ = self.rnn(features); return self.linear(rnn_out)

# --- B. CRNN (GRU) ---
class CRNN_GRU(nn.Module):
    def __init__(self, num_chars):
        super(CRNN_GRU, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.MaxPool2d(2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1), nn.MaxPool2d(2, 2), nn.ReLU(), nn.Conv2d(64, 128, 3, padding=1), nn.MaxPool2d((2, 1)), nn.ReLU(), nn.BatchNorm2d(128))
        self.rnn = nn.GRU(512, 64, bidirectional=True, batch_first=True); self.linear = nn.Linear(128, num_chars + 1)
    def forward(self, x):
        features = self.cnn(x); features = features.permute(0, 3, 1, 2).view(features.size(0), features.size(3), -1)
        rnn_out, _ = self.rnn(features); return self.linear(rnn_out)

# --- C. RESNET-GRU ---
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False); self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes: self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes))
    def forward(self, x): return torch.relu(self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class ResNet_GRU(nn.Module):
    def __init__(self, num_chars):
        super(ResNet_GRU, self).__init__()
        self.in_planes = 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False); self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 2, stride=2); self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=(2, 1)); self.layer4 = self._make_layer(256, 2, stride=(2, 1))
        self.rnn = nn.GRU(512, 64, bidirectional=True, batch_first=True); self.linear = nn.Linear(128, num_chars + 1)
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1); layers = []
        for stride in strides: layers.append(BasicBlock(self.in_planes, planes, stride)); self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x))); out = self.layer1(out); out = self.layer2(out); out = self.layer3(out); out = self.layer4(out)
        out = out.permute(0, 3, 1, 2).reshape(out.size(0), out.size(3), -1); out, _ = self.rnn(out)
        return self.linear(out)

# --- D. TRANSFORMER ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model); position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term); self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:x.size(0), :].unsqueeze(1)

class OCRTransformer(nn.Module):
    def __init__(self, num_chars):
        super(OCRTransformer, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.MaxPool2d(2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1), nn.MaxPool2d(2, 2), nn.ReLU(), nn.Conv2d(64, 128, 3, padding=1), nn.MaxPool2d((2, 1)), nn.ReLU(), nn.BatchNorm2d(128))
        self.d_model = 128; self.pos_encoder = PositionalEncoding(d_model=self.d_model)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=512, dropout=0.1), num_layers=2)
        self.projection = nn.Linear(512, self.d_model); self.linear = nn.Linear(self.d_model, num_chars + 1)
    def forward(self, x):
        features = self.cnn(x); features = features.permute(3, 0, 1, 2).view(features.size(3), features.size(0), -1)
        return self.linear(self.transformer_encoder(self.pos_encoder(self.projection(features))))

# --- E. MOBILENET-GRU ---
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False); self.bn = nn.BatchNorm2d(out_channels); self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.bn(self.pointwise(self.depthwise(x))))

class MobileNet_GRU(nn.Module):
    def __init__(self, num_chars):
        super(MobileNet_GRU, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False); self.bn1 = nn.BatchNorm2d(32); self.relu = nn.ReLU(inplace=True)
        self.features = nn.Sequential(DepthwiseSeparableConv(32, 64, stride=2), DepthwiseSeparableConv(64, 64, stride=1), DepthwiseSeparableConv(64, 128, stride=2), DepthwiseSeparableConv(128, 128, stride=1), DepthwiseSeparableConv(128, 128, stride=1), nn.MaxPool2d((2, 1)), DepthwiseSeparableConv(128, 256, stride=1), nn.MaxPool2d((2, 1)), DepthwiseSeparableConv(256, 256, stride=1))
        self.rnn = nn.GRU(512, 64, bidirectional=True, batch_first=True); self.linear = nn.Linear(128, num_chars + 1)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x))); x = self.features(x); x = x.permute(0, 3, 1, 2).reshape(x.size(0), x.size(3), -1); x, _ = self.rnn(x)
        return self.linear(x)

# --- F. VGG-16 ---
class VGG_BiLSTM(nn.Module):
    def __init__(self, num_chars):
        super(VGG_BiLSTM, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True), nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1)), nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True), nn.BatchNorm2d(512), nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.BatchNorm2d(512), nn.MaxPool2d((2, 1)), nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(True))
        self.rnn = nn.LSTM(512, 256, bidirectional=True, batch_first=True); self.linear = nn.Linear(512, num_chars + 1)
    def forward(self, x):
        conv = self.features(x); conv = conv.squeeze(2).permute(0, 2, 1); rnn_out, _ = self.rnn(conv)
        return self.linear(rnn_out)

# --- G. DENSENET-121 ---
class DenseNet_GRU(nn.Module):
    def __init__(self, num_chars):
        super(DenseNet_GRU, self).__init__()
        self.densenet = models.densenet121(weights=None); self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False); self.densenet.features.pool0 = nn.Identity(); self.densenet.features.transition3.pool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.adapter = nn.Linear(4096, 512); self.rnn = nn.GRU(512, 64, bidirectional=True, batch_first=True); self.linear = nn.Linear(128, num_chars + 1)
    def forward(self, x):
        features = self.densenet.features(x); features = features.permute(0, 3, 1, 2).reshape(features.size(0), features.size(3), -1); features = self.adapter(features); out, _ = self.rnn(features)
        return self.linear(out)

# --- H. EFFICIENTNET-B0 ---
class EfficientNet_GRU(nn.Module):
    def __init__(self, num_chars):
        super(EfficientNet_GRU, self).__init__()
        self.effnet = models.efficientnet_b0(weights=None); self.effnet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=(2, 1), padding=1, bias=False); self.effnet.features[2][0].block[1][0].stride = (2, 1); self.effnet.features[3][0].block[1][0].stride = (2, 1); self.effnet.features[4][0].block[1][0].stride = (2, 1); self.effnet.features[6][0].block[1][0].stride = (1, 1)
        self.adapter = nn.Linear(2560, 512); self.rnn = nn.GRU(512, 64, bidirectional=True, batch_first=True); self.linear = nn.Linear(128, num_chars + 1)
    def forward(self, x):
        features = self.effnet.features(x); features = features.permute(0, 3, 1, 2).reshape(features.size(0), features.size(3), -1); features = self.adapter(features); out, _ = self.rnn(features)
        return self.linear(out)

# =============================================================================
# 2. NLP MODELS
# =============================================================================
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size; self.embedding = nn.Embedding(input_size, hidden_size); self.gru = nn.GRU(hidden_size, hidden_size)
    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1); output, hidden = self.gru(output, hidden); return output, hidden
    def initHidden(self): return torch.zeros(1, 1, self.hidden_size).to(DEVICE)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size; self.embedding = nn.Embedding(output_size, hidden_size); self.gru = nn.GRU(hidden_size, hidden_size); self.out = nn.Linear(hidden_size, output_size); self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1); output = torch.relu(output); output, hidden = self.gru(output, hidden); output = self.softmax(self.out(output[0])); return output, hidden

# =============================================================================
# 3. EVALUATION UTILS
# =============================================================================
def get_vocab():
    if not os.path.exists(TRAIN_DIR): return 0, {}
    chars = set(); files = [f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(('.jpg', '.png'))]
    for f in files:
        for c in f.split('_')[0]: chars.add(c)
    chars = sorted(list(chars)); return len(chars), {i+1: c for i, c in enumerate(chars)}

def decode(output, idx2char):
    arg_maxes = torch.argmax(output, dim=2); decodes = []
    for i in range(arg_maxes.size(1)):
        seq = arg_maxes[:, i].tolist(); res = []; prev = 0
        for idx in seq:
            if idx != 0 and idx != prev: res.append(idx2char[idx])
            prev = idx
        decodes.append(''.join(res))
    return decodes[0]

def run_nlp_correction(raw_text, encoder, decoder, char2idx_nlp, idx2char_nlp):
    for c in raw_text: 
        if c not in char2idx_nlp: return raw_text
    input_tensor = torch.tensor([char2idx_nlp[c] for c in raw_text], dtype=torch.long, device=DEVICE).view(-1, 1)
    encoder_hidden = encoder.initHidden()
    for ei in range(input_tensor.size(0)): _, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
    decoder_input = torch.tensor([[char2idx_nlp['<SOS>']]], device=DEVICE)
    decoder_hidden = encoder_hidden
    decoded_chars = []
    for _ in range(20):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden); topv, topi = decoder_output.topk(1); char_idx = topi.item()
        if char_idx == char2idx_nlp['<EOS>']: break
        decoded_chars.append(idx2char_nlp[char_idx]); decoder_input = topi.squeeze().detach()
    return "".join(decoded_chars)

def evaluate_single_model(model_path, model_name, device, transform, num_chars, idx2char, encoder, decoder, char2idx_nlp, idx2char_nlp):
    name = model_name.lower()
    # Architecture Switching
    if "efficientnet" in name: model = EfficientNet_GRU(num_chars).to(device)
    elif "densenet" in name: model = DenseNet_GRU(num_chars).to(device)
    elif "resnet" in name: model = ResNet_GRU(num_chars).to(device)
    elif "mobilenet" in name: model = MobileNet_GRU(num_chars).to(device)
    elif "transformer" in name: model = OCRTransformer(num_chars).to(device)
    elif "vgg" in name: model = VGG_BiLSTM(num_chars).to(device)
    elif "gru" in name: model = CRNN_GRU(num_chars).to(device)
    else: model = CRNN(num_chars).to(device)

    try: model.load_state_dict(torch.load(model_path, map_location=device)); model.eval()
    except: return 0, 0, 0, 0, 0

    files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.png'))]
    if not files: return 0, 0, 0, 0, 0

    raw_correct, sys_correct, nlp_correct = 0, 0, 0
    times = []

    with torch.no_grad():
        for f in files:
            img_path = os.path.join(TEST_DIR, f)
            try:
                image = Image.open(img_path).convert('L')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                start = time.time()
                output = model(image_tensor)
                # Special handling for Transformer
                if "Transformer" not in str(type(model)):
                    output = output.permute(1, 0, 2)
                
                raw_pred = decode(output, idx2char)
                end = time.time()
                times.append((end - start) * 1000)

                # 1. Heuristic Check
                matches = difflib.get_close_matches(raw_pred, MEDICINE_DB, n=1, cutoff=0.1)
                sys_pred = matches[0] if matches else "Unknown"

                # 2. NLP Check
                if encoder: nlp_pred = run_nlp_correction(raw_pred, encoder, decoder, char2idx_nlp, idx2char_nlp)
                else: nlp_pred = "N/A"

                real = f.split('_')[0].split('.')[0]
                if raw_pred.lower() == real.lower(): raw_correct += 1
                if sys_pred.lower() == real.lower(): sys_correct += 1
                if nlp_pred.lower() == real.lower(): nlp_correct += 1
            except: continue

    avg_time = sum(times) / len(times) if times else 0
    model_size = os.path.getsize(model_path) / (1024 * 1024) # MB
    return (raw_correct/len(files))*100, (sys_correct/len(files))*100, (nlp_correct/len(files))*100, avg_time, model_size

# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================
def main():
    print("--- ULTIMATE BENCHMARK (8 MODELS + 3 METRICS) ---")
    print(f"Scanning: {os.path.abspath(MODELS_ROOT_DIR)}")
    
    # Load Data
    num_chars, idx2char = get_vocab()
    if num_chars == 0: return

    # Load NLP
    char2idx_nlp, idx2char_nlp, encoder, decoder = None, None, None, None
    if os.path.exists(NLP_MODEL_PATH):
        print("[INFO] Loading NLP Corrector...")
        checkpoint = torch.load(NLP_MODEL_PATH, map_location=DEVICE)
        nlp_chars = checkpoint['chars']
        char2idx_nlp = {c: i for i, c in enumerate(nlp_chars)}
        idx2char_nlp = {i: c for i, c in enumerate(nlp_chars)}
        encoder = EncoderRNN(len(nlp_chars), 128).to(DEVICE)
        decoder = DecoderRNN(128, len(nlp_chars)).to(DEVICE)
        encoder.load_state_dict(checkpoint['enc'])
        decoder.load_state_dict(checkpoint['dec'])
        encoder.eval(); decoder.eval()
    else:
        print("[WARN] NLP Corrector NOT found.")

    transform = transforms.Compose([transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    subfolders = [f.path for f in os.scandir(MODELS_ROOT_DIR) if f.is_dir()]
    
    data = []
    print(f"\n{'MODEL':<20} | {'RAW':<7} | {'SYSTEM':<7} | {'NEURAL':<7} | {'SPEED':<7} | {'SIZE':<7}")
    print("-" * 80)

    for folder in subfolders:
        model_file = os.path.join(folder, 'best_model.pth')
        if os.path.exists(model_file):
            name = os.path.basename(folder)
            raw, sys, nlp, speed, size = evaluate_single_model(model_file, name, DEVICE, transform, num_chars, idx2char, encoder, decoder, char2idx_nlp, idx2char_nlp)
            
            if raw > 0 or sys > 0:
                print(f"{name:<20} | {raw:<7.1f} | {sys:<7.1f} | {nlp:<7.1f} | {speed:<7.1f} | {size:<7.1f}")
                data.append({'Model': name, 'Raw': raw, 'System': sys, 'Neural': nlp, 'Speed': speed, 'Size': size})

    if not data: return

    # --- PLOTTING ---
    df = pd.DataFrame(data)
    
    # 1. ACCURACY CHART (3 Bars)
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(df['Model'])); width = 0.25
    
    r1 = ax.bar(x - width, df['Raw'], width, label='Raw Vision', color='#95a5a6')
    r2 = ax.bar(x, df['System'], width, label='Hybrid System', color='#27ae60')
    r3 = ax.bar(x + width, df['Neural'], width, label='Neural NLP', color='#8e44ad')
    
    ax.set_ylabel('Accuracy (%)'); ax.set_title('Final Accuracy Comparison'); ax.set_xticks(x); ax.set_xticklabels(df['Model'], rotation=20); ax.legend()
    ax.bar_label(r1, fmt='%.1f'); ax.bar_label(r2, fmt='%.1f'); ax.bar_label(r3, fmt='%.1f')
    plt.tight_layout(); plt.savefig('final_accuracy_chart.png')

    # 2. EFFICIENCY CHART
    plt.figure(figsize=(10, 6))
    for i, row in df.iterrows():
        plt.scatter(row['Speed'], row['System'], s=row['Size']*15, label=row['Model'], alpha=0.7)
        plt.text(row['Speed'], row['System'], f"{row['Model']}\n{row['Size']:.1f}MB", fontsize=8)
    plt.xlabel('Speed (ms)'); plt.ylabel('Accuracy (%)'); plt.title('Efficiency Frontier'); plt.grid(True, linestyle='--'); plt.tight_layout()
    plt.savefig('final_efficiency_chart.png')
    
    print("\nDone! Charts saved.")

if __name__ == "__main__":
    main()