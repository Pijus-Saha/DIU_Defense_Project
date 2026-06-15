import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import random
import time

# --- CONFIG ---
VISION_MODEL_PATH = '../EfficientNet_GRU_v8/best_model.pth'
DATA_DIR = '../../02_Data_Processor/labeled_dataset_kaggle'
HIDDEN_SIZE = 128
LR = 0.001
EPOCHS = 100 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Medicine Database (Source of Truth)
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

# --- 1. VISION MODEL ---
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

# --- 2. NLP MODEL ---
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

# --- 3. UTILS & GENERATORS ---
def get_vocab():
    # Build vocab from Medicine DB + Common OCR noise chars
    chars = set()
    for m in MEDICINE_DB:
        for c in m: chars.add(c)
    
    # Add noise characters often seen in OCR errors
    chars.update(['@', '0', '1', '3', '$', '(', ')', '[', ']', '|', '!', '?', '_'])
    
    chars.add('<SOS>')
    chars.add('<EOS>')
    chars = sorted(list(chars))
    return chars, {c: i for i, c in enumerate(chars)}, {i: c for i, c in enumerate(chars)}

def add_synthetic_noise(word):
    # Simulate typical OCR failures
    chars = list(word)
    new_chars = []
    for c in chars:
        if random.random() < 0.15: # 15% chance of error
            if c in 'lL1I|': new_chars.append(random.choice(['1', 'I', '|', 'l']))
            elif c in 'a@q': new_chars.append(random.choice(['@', 'q', 'o']))
            elif c in 'eEc': new_chars.append(random.choice(['c', 'o', 'e']))
            elif c in 'oO0': new_chars.append('0')
            elif c in 'sS5': new_chars.append('5')
            elif c in 'tT+': new_chars.append('+')
            else: new_chars.append(c) # Keep original
        else:
            new_chars.append(c)
            
    # Randomly delete a char (missed detection)
    if len(new_chars) > 3 and random.random() < 0.1:
        del new_chars[random.randint(0, len(new_chars)-1)]
        
    return "".join(new_chars)

def decode_vision(output, idx2char):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    for i in range(arg_maxes.size(1)):
        seq = arg_maxes[:, i].tolist()
        res = []
        prev = 0
        for idx in seq:
            if idx != 0 and idx != prev: res.append(idx2char[idx])
            prev = idx
        decodes.append(''.join(res))
    return decodes[0]

def tensorFromSentence(char2idx, sentence):
    indexes = [char2idx[c] for c in sentence if c in char2idx]
    indexes.append(char2idx['<EOS>'])
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)

# --- 4. MAIN ---
def main():
    print("--- ROBUST NLP TRAINING (Real + Synthetic) ---")
    
    chars, char2idx, idx2char = get_vocab()
    n_chars = len(chars)
    print(f"Vocab Size: {n_chars}")

    training_data = []

    # A. Generate SYNTHETIC Data (Massive amount)
    print("Generating Synthetic Errors...")
    for _ in range(5000): # 5000 fake samples
        real = random.choice(MEDICINE_DB)
        fake = add_synthetic_noise(real)
        if fake != real:
            training_data.append((fake, real))
            
    # B. Generate REAL Data (From Vision Model)
    # (Optional: If vision model fails to load, we still have synthetic data)
    try:
        # Load Vision Vocab from file scan
        v_chars = set()
        for f in os.listdir(DATA_DIR):
            if f.endswith('.jpg'):
                for c in f.split('_')[0]: v_chars.add(c)
        v_idx2char = {i+1: c for i, c in enumerate(sorted(list(v_chars)))}
        
        vision_model = EfficientNet_GRU(len(v_chars)).to(DEVICE)
        vision_model.load_state_dict(torch.load(VISION_MODEL_PATH, map_location=DEVICE))
        vision_model.eval()
        
        print("Vision Model Loaded. Mining real errors...")
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')]
        transform = transforms.Compose([transforms.Resize((32, 128)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        
        real_errors = 0
        with torch.no_grad():
            for i, f in enumerate(files):
                if i > 500: break # Scan 500 images
                real_label = f.split('_')[0]
                img = Image.open(os.path.join(DATA_DIR, f)).convert('L')
                tensor = transform(img).unsqueeze(0).to(DEVICE)
                output = vision_model(tensor).permute(1, 0, 2)
                pred = decode_vision(output, v_idx2char)
                
                # Add ALL predictions (even correct ones, to reinforce stability)
                training_data.append((pred, real_label))
                if pred != real_label: real_errors += 1
                
        print(f"Added {len(training_data)} samples total (Real Errors Found: {real_errors})")
        
    except Exception as e:
        print(f"Warning: Could not load vision model ({e}). Using Synthetic Data Only.")

    # C. Train
    encoder = EncoderRNN(n_chars, HIDDEN_SIZE).to(DEVICE)
    decoder = DecoderRNN(HIDDEN_SIZE, n_chars).to(DEVICE)
    enc_optimizer = optim.Adam(encoder.parameters(), lr=LR)
    dec_optimizer = optim.Adam(decoder.parameters(), lr=LR)
    criterion = nn.NLLLoss()

    print(f"Training on {len(training_data)} pairs...")
    
    for epoch in range(1, EPOCHS + 1):
        random.shuffle(training_data)
        total_loss = 0
        
        for input_text, target_text in training_data:
            if len(input_text) == 0: continue
            
            input_tensor = tensorFromSentence(char2idx, input_text)
            target_tensor = tensorFromSentence(char2idx, target_text)

            encoder_hidden = encoder.initHidden()
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            input_len = input_tensor.size(0)
            target_len = target_tensor.size(0)
            loss = 0

            for ei in range(input_len):
                _, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

            decoder_input = torch.tensor([[char2idx['<SOS>']]], device=DEVICE)
            decoder_hidden = encoder_hidden

            for di in range(target_len):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]

            loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            total_loss += loss.item() / target_len

        if epoch % 10 == 0:
            avg_loss = total_loss / len(training_data)
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
            # Show a test case
            test_word = "Az1thr0"
            print(f"   Test: {test_word} -> {evaluate_test(encoder, decoder, char2idx, idx2char, test_word)}")

    torch.save({'enc': encoder.state_dict(), 'dec': decoder.state_dict(), 'chars': chars}, 'nlp_corrector.pth')
    print("Saved 'nlp_corrector.pth'")

def evaluate_test(encoder, decoder, char2idx, idx2char, word):
    with torch.no_grad():
        input_tensor = tensorFromSentence(char2idx, word)
        encoder_hidden = encoder.initHidden()
        for ei in range(input_tensor.size(0)):
            _, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        
        decoder_input = torch.tensor([[char2idx['<SOS>']]], device=DEVICE)
        decoder_hidden = encoder_hidden
        decoded_chars = []
        
        for _ in range(20):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            if topi.item() == char2idx['<EOS>']: break
            decoded_chars.append(idx2char[topi.item()])
            decoder_input = topi.squeeze().detach()
            
        return "".join(decoded_chars)

if __name__ == "__main__":
    main()