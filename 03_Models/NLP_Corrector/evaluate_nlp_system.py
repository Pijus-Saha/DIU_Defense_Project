import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import difflib
import sys

# --- CONFIGURATION ---
# 1. Vision Model Path (EfficientNet)
VISION_MODEL_PATH = '../EfficientNet_GRU/best_model.pth'
# 2. NLP Model Path
NLP_MODEL_PATH = 'nlp_corrector.pth'

# Datasets
TRAIN_DATA_DIR = '../../02_Data_Processor/labeled_dataset_kaggle' 
TEST_DATA_DIR = '../../02_Data_Processor/labeled_dataset_test' 

IMG_HEIGHT = 32
IMG_WIDTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Medicine DB for final verification step
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
# 1. VISION MODEL DEFINITION
# ==========================================
class EfficientNet_GRU(nn.Module):
    def __init__(self, num_chars):
        super(EfficientNet_GRU, self).__init__()
        self.effnet = models.efficientnet_b0(weights=None)
        self.effnet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.effnet.features[2][0].block[1][0].stride = (2, 1)
        self.effnet.features[3][0].block[1][0].stride = (2, 1)
        self.effnet.features[4][0].block[1][0].stride = (2, 1)
        self.effnet.features[6][0].block[1][0].stride = (1, 1)
        self.adapter = nn.Linear(2560, 512) 
        self.rnn = nn.GRU(512, 64, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(128, num_chars + 1)

    def forward(self, x):
        features = self.effnet.features(x)
        b, c, h, w = features.size()
        features = features.permute(0, 3, 1, 2).reshape(b, w, c * h)
        features = self.adapter(features)
        out, _ = self.rnn(features)
        return self.linear(out)

# ==========================================
# 2. NLP MODEL DEFINITIONS
# ==========================================
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size).to(DEVICE)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

# ==========================================
# 3. UTILS
# ==========================================
def get_vision_vocab():
    if not os.path.exists(TRAIN_DATA_DIR): return 0, {}
    chars = set()
    files = [f for f in os.listdir(TRAIN_DATA_DIR) if f.lower().endswith(('.jpg', '.png'))]
    for f in files:
        for c in f.split('_')[0]: chars.add(c)
    chars = sorted(list(chars))
    return len(chars), {i+1: c for i, c in enumerate(chars)}

def decode_vision(output, idx2char):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    for i in range(arg_maxes.size(1)):
        seq = arg_maxes[:, i].tolist(); res = []; prev = 0
        for idx in seq:
            if idx != 0 and idx != prev: res.append(idx2char[idx])
            prev = idx
        decodes.append(''.join(res))
    return decodes[0]

def run_nlp_correction(raw_text, encoder, decoder, char2idx, idx2char):
    # Safety: check input chars
    for c in raw_text:
        if c not in char2idx: return raw_text
        
    input_tensor = torch.tensor([char2idx[c] for c in raw_text], dtype=torch.long, device=DEVICE).view(-1, 1)
    
    encoder_hidden = encoder.initHidden()
    for ei in range(input_tensor.size(0)):
        _, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
        
    decoder_input = torch.tensor([[char2idx['<SOS>']]], device=DEVICE)
    decoder_hidden = encoder_hidden
    
    decoded_chars = []
    for _ in range(20):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        char_idx = topi.item()
        if char_idx == char2idx['<EOS>']: break
        decoded_chars.append(idx2char[char_idx])
        decoder_input = topi.squeeze().detach()
        
    return "".join(decoded_chars)

# ==========================================
# 4. MAIN EVALUATION
# ==========================================
def main():
    print("--- EVALUATING FULL SYSTEM (VISION + NLP) ---")
    
    # A. Load Vision Model
    num_chars_vis, idx2char_vis = get_vision_vocab()
    vision_model = EfficientNet_GRU(num_chars_vis).to(DEVICE)
    try:
        vision_model.load_state_dict(torch.load(VISION_MODEL_PATH, map_location=DEVICE))
        vision_model.eval()
        print("Vision Model Loaded.")
    except Exception as e:
        print(f"Error loading Vision Model: {e}")
        return

    # B. Load NLP Model
    try:
        checkpoint = torch.load(NLP_MODEL_PATH, map_location=DEVICE)
        nlp_chars = checkpoint['chars']
        char2idx_nlp = {c: i for i, c in enumerate(nlp_chars)}
        idx2char_nlp = {i: c for i, c in enumerate(nlp_chars)}
        
        encoder = EncoderRNN(len(nlp_chars), 128).to(DEVICE)
        decoder = DecoderRNN(128, len(nlp_chars)).to(DEVICE)
        
        encoder.load_state_dict(checkpoint['enc'])
        decoder.load_state_dict(checkpoint['dec'])
        encoder.eval(); decoder.eval()
        print("NLP Model Loaded.")
    except Exception as e:
        print(f"Error loading NLP Model: {e}")
        return

    # C. Evaluate
    files = [f for f in os.listdir(TEST_DATA_DIR) if f.lower().endswith(('.jpg', '.png'))]
    total = len(files)
    print(f"Testing {total} images...")
    
    transform = transforms.Compose([transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    count_pass = 0      # Vision correct
    count_sys_pass = 0  # Vision wrong, NLP fixed it
    count_ai_pass = 0   # Vision correct, NLP broke it (Bad!)
    count_fail = 0      # Both wrong
    
    with torch.no_grad():
        for i, f in enumerate(files):
            real_label = f.split('_')[0].split('.')[0]
            img_path = os.path.join(TEST_DATA_DIR, f)
            
            try:
                # 1. Vision
                image = Image.open(img_path).convert('L')
                image_tensor = transform(image).unsqueeze(0).to(DEVICE)
                output = vision_model(image_tensor).permute(1, 0, 2)
                raw_pred = decode_vision(output, idx2char_vis)
                
                # 2. NLP Correction
                nlp_pred = run_nlp_correction(raw_pred, encoder, decoder, char2idx_nlp, idx2char_nlp)
                
                # 3. Hybrid Safety Net (Optional: Verify against DB)
                # If NLP predicts something NOT in DB, use raw if better
                final_pred = nlp_pred
                if final_pred not in MEDICINE_DB:
                     # Fallback logic: find closest in DB to what NLP said
                     matches = difflib.get_close_matches(nlp_pred, MEDICINE_DB, n=1, cutoff=0.5)
                     if matches: final_pred = matches[0]

                # 4. Scoring
                is_vision_correct = (raw_pred.lower() == real_label.lower())
                is_final_correct = (final_pred.lower() == real_label.lower())
                
                if is_vision_correct and is_final_correct:
                    count_pass += 1
                elif not is_vision_correct and is_final_correct:
                    count_sys_pass += 1
                    # print(f"Saved: {raw_pred} -> {final_pred} (Real: {real_label})")
                elif is_vision_correct and not is_final_correct:
                    count_ai_pass += 1
                    # print(f"Broke: {raw_pred} -> {final_pred} (Real: {real_label})")
                else:
                    count_fail += 1
                    
            except: pass

    # --- REPORT ---
    raw_acc = ((count_pass + count_ai_pass) / total) * 100
    final_acc = ((count_pass + count_sys_pass) / total) * 100
    
    print("\n" + "="*50)
    print("      MULTIMODAL SYSTEM EVALUATION")
    print("="*50)
    print(f"Raw Vision Accuracy:     {raw_acc:.2f}%")
    print(f"Final System Accuracy:   {final_acc:.2f}%")
    print("-" * 50)
    print(f"Pure Vision Correct:     {count_pass}")
    print(f"NLP Recovered Errors:    {count_sys_pass} (Value Add)")
    print(f"NLP Introduced Errors:   {count_ai_pass} (Regression)")
    print(f"Total Failures:          {count_fail}")
    print("="*50)

if __name__ == "__main__":
    main()