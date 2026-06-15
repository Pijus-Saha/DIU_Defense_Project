import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import difflib
import cv2
import numpy as np

# --- CONFIGURATION ---
# 1. VISUAL MODEL (The Champion: EfficientNet)
VISION_MODEL_PATH = '../03_Models/EfficientNet_GRU_v8/best_model.pth'
# 2. SPELL CHECKER MODEL (The Fixer: NLP)
NLP_MODEL_PATH = '../03_Models/NLP_Corrector/nlp_corrector.pth'

DATA_DIR = '../02_Data_Processor/labeled_dataset_kaggle'
IMG_HEIGHT = 32
IMG_WIDTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- THEME COLORS ---
COLOR_PRIMARY = "#2C3E50"    # Dark Blue
COLOR_SECONDARY = "#1ABC9C"  # Teal
COLOR_BG = "#ECF0F1"         # Light Gray
COLOR_WHITE = "#FFFFFF"
COLOR_TEXT = "#34495E"

# --- FULL MEDICINE DATABASE (Required for final verification) ---
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
# 1. VISION MODEL: EFFICIENTNET-GRU
# ==========================================
class EfficientNet_GRU(nn.Module):
    def __init__(self, num_chars):
        super(EfficientNet_GRU, self).__init__()
        self.effnet = models.efficientnet_b0(weights=None)
        # Modify Stem
        self.effnet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        # Stride Hacks to preserve width
        self.effnet.features[2][0].block[1][0].stride = (2, 1)
        self.effnet.features[3][0].block[1][0].stride = (2, 1)
        self.effnet.features[4][0].block[1][0].stride = (2, 1)
        self.effnet.features[6][0].block[1][0].stride = (1, 1)
        # Adapter
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
# 2. NLP MODEL: SEQ2SEQ (The Fixer)
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
# 3. GUI APPLICATION
# ==========================================
class ModernApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical OCR System v10.0 (EfficientNet + Hybrid Logic)")
        self.root.geometry("900x600")
        self.root.configure(bg=COLOR_BG)
        
        self.current_cv_image = None
        self.nlp_chars = None
        
        self.init_models()
        self.create_header()
        self.create_main_area()
        self.create_footer()

    def init_models(self):
        # 1. Load Vision Model
        try:
            self.vocab_chars, self.idx2char_vis = self.get_vision_vocab()
            self.vision_model = EfficientNet_GRU(len(self.vocab_chars)).to(DEVICE)
            self.vision_model.load_state_dict(torch.load(VISION_MODEL_PATH, map_location=DEVICE))
            self.vision_model.eval()
            print("Vision Model Loaded.")
        except Exception as e:
            messagebox.showerror("Error", f"Vision Model Failed: {e}")

        # 2. Load NLP Model
        try:
            if os.path.exists(NLP_MODEL_PATH):
                checkpoint = torch.load(NLP_MODEL_PATH, map_location=DEVICE)
                self.nlp_chars = checkpoint['chars']
                self.char2idx_nlp = {c: i for i, c in enumerate(self.nlp_chars)}
                self.idx2char_nlp = {i: c for i, c in enumerate(self.nlp_chars)}
                
                self.encoder = EncoderRNN(len(self.nlp_chars), 128).to(DEVICE)
                self.decoder = DecoderRNN(128, len(self.nlp_chars)).to(DEVICE)
                
                self.encoder.load_state_dict(checkpoint['enc'])
                self.decoder.load_state_dict(checkpoint['dec'])
                self.encoder.eval()
                self.decoder.eval()
                print("NLP Model Loaded.")
            else:
                print("NLP Model not found! Falling back to logic.")
                self.nlp_chars = None
        except Exception as e:
            print(f"NLP Model Error: {e}")
            self.nlp_chars = None

    def create_header(self):
        header = tk.Frame(self.root, bg=COLOR_PRIMARY, height=80)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        try:
            logo_img = Image.open("assets/logo.png")
            logo_img = logo_img.resize((60, 60), Image.Resampling.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            lbl_logo = tk.Label(header, image=self.logo_photo, bg=COLOR_PRIMARY)
            lbl_logo.pack(side="left", padx=20)
        except:
            lbl_logo = tk.Label(header, text="[FYP]", bg=COLOR_PRIMARY, fg="white", font=("Arial", 14, "bold"))
            lbl_logo.pack(side="left", padx=20)

        title_frame = tk.Frame(header, bg=COLOR_PRIMARY)
        title_frame.pack(side="left", pady=10)
        
        lbl_title = tk.Label(title_frame, text="AI Prescription Analyzer", font=("Helvetica", 18, "bold"), fg="white", bg=COLOR_PRIMARY)
        lbl_title.pack(anchor="w")
        lbl_subtitle = tk.Label(title_frame, text="Powered by EfficientNet & Algorithmic Correction", font=("Helvetica", 10), fg="#BDC3C7", bg=COLOR_PRIMARY)
        lbl_subtitle.pack(anchor="w")

    def create_main_area(self):
        container = tk.Frame(self.root, bg=COLOR_BG)
        container.pack(fill="both", expand=True, padx=20, pady=20)

        # --- LEFT PANEL ---
        left_panel = tk.Frame(container, bg=COLOR_WHITE, width=400, highlightthickness=1, highlightbackground="#BDC3C7")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        left_panel.pack_propagate(False)

        tk.Label(left_panel, text="Input Image", font=("Helvetica", 12, "bold"), bg=COLOR_WHITE, fg=COLOR_TEXT).pack(pady=15)

        # FIXED IMAGE CONTAINER
        self.img_container = tk.Frame(left_panel, bg="#F0F3F4", width=360, height=250)
        self.img_container.pack(padx=20, pady=10)
        self.img_container.pack_propagate(False) 

        self.panel = tk.Label(self.img_container, bg="#F0F3F4", text="No Image Selected", fg="#95A5A6")
        self.panel.pack(expand=True)

        # Buttons
        btn_frame = tk.Frame(left_panel, bg=COLOR_WHITE)
        btn_frame.pack(pady=20)

        self.btn_upload = tk.Button(btn_frame, text="📂 Upload Image", command=self.load_image, 
                                    bg=COLOR_SECONDARY, fg="white", font=("Helvetica", 11, "bold"), 
                                    relief="flat", padx=15, pady=8, cursor="hand2")
        self.btn_upload.pack(side="left", padx=5)

        self.btn_clear = tk.Button(btn_frame, text="↺ Reset", command=self.reset_ui, 
                                   bg="#E74C3C", fg="white", font=("Helvetica", 11, "bold"), 
                                   relief="flat", padx=15, pady=8, cursor="hand2")
        self.btn_clear.pack(side="left", padx=5)

        # --- RIGHT PANEL ---
        right_panel = tk.Frame(container, bg=COLOR_WHITE, width=400, highlightthickness=1, highlightbackground="#BDC3C7")
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))
        right_panel.pack_propagate(False)

        tk.Label(right_panel, text="Analysis Results", font=("Helvetica", 12, "bold"), bg=COLOR_WHITE, fg=COLOR_TEXT).pack(pady=15)

        self.create_result_card(right_panel, "Raw Vision Prediction", "lbl_raw", "gray")
        tk.Frame(right_panel, bg=COLOR_WHITE, height=20).pack()
        self.create_result_card(right_panel, "Corrected Output", "lbl_nlp", "#2980B9", font_size=20)

        conf_frame = tk.Frame(right_panel, bg=COLOR_WHITE)
        conf_frame.pack(fill="x", padx=30, pady=30)
        tk.Label(conf_frame, text="System Confidence Score", bg=COLOR_WHITE, fg=COLOR_TEXT).pack(anchor="w")
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(conf_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=5)
        
        self.lbl_conf_text = tk.Label(conf_frame, text="0%", bg=COLOR_WHITE, fg="gray", font=("Arial", 10))
        self.lbl_conf_text.pack(anchor="e")

    def create_result_card(self, parent, title, attr_name, color, font_size=12):
        frame = tk.Frame(parent, bg="#F7F9F9", highlightthickness=1, highlightbackground="#E5E8E8")
        frame.pack(fill="x", padx=30)
        tk.Label(frame, text=title, bg="#F7F9F9", fg="#7F8C8D", font=("Arial", 9)).pack(anchor="w", padx=10, pady=(10, 0))
        lbl = tk.Label(frame, text="---", bg="#F7F9F9", fg=color, font=("Helvetica", font_size, "bold"))
        lbl.pack(anchor="w", padx=10, pady=(5, 15))
        setattr(self, attr_name, lbl)

    def create_footer(self):
        footer = tk.Frame(self.root, bg="#BDC3C7", height=30)
        footer.pack(fill="x", side="bottom")
        tk.Label(footer, text="Final Year Project 2025 | CSE Dept", bg="#BDC3C7", fg="white", font=("Arial", 8)).pack(pady=5)

    # --- LOGIC ---
    def get_vision_vocab(self):
        chars = set()
        for f in os.listdir(DATA_DIR):
            if f.lower().endswith(('.jpg', '.png')):
                for c in f.split('_')[0]: chars.add(c)
        chars = sorted(list(chars))
        return chars, {i+1: c for i, c in enumerate(chars)}

    def decode_vision(self, output):
        arg_maxes = torch.argmax(output, dim=2)
        decodes = []
        for i in range(arg_maxes.size(1)):
            seq = arg_maxes[:, i].tolist(); res = []; prev = 0
            for idx in seq:
                if idx != 0 and idx != prev: res.append(self.idx2char_vis[idx])
                prev = idx
            decodes.append(''.join(res))
        return decodes[0]

    def run_nlp_correction(self, raw_text):
        if self.nlp_chars is None: return raw_text 
        
        # Check if characters exist in NLP vocab
        for c in raw_text:
            if c not in self.char2idx_nlp: return raw_text 
            
        input_tensor = torch.tensor([self.char2idx_nlp[c] for c in raw_text], dtype=torch.long, device=DEVICE).view(-1, 1)
        
        encoder_hidden = self.encoder.initHidden()
        for ei in range(input_tensor.size(0)):
            _, encoder_hidden = self.encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
            
        decoder_input = torch.tensor([[self.char2idx_nlp['<SOS>']]], device=DEVICE)
        decoder_hidden = encoder_hidden
        
        decoded_chars = []
        for _ in range(20):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            char_idx = topi.item()
            
            if char_idx == self.char2idx_nlp['<EOS>']: break
            
            decoded_chars.append(self.idx2char_nlp[char_idx])
            decoder_input = topi.squeeze().detach()
            
        return "".join(decoded_chars)

    def reset_ui(self):
        self.panel.configure(image='', text="No Image Selected")
        self.lbl_raw.config(text="---")
        self.lbl_nlp.config(text="---", fg="#2980B9")
        self.progress_var.set(0)
        self.lbl_conf_text.config(text="0%")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png")])
        if not path: return

        # Display Image
        img = Image.open(path)
        display_max_width = 350
        display_max_height = 240
        img_display = img.copy()
        img_display.thumbnail((display_max_width, display_max_height), Image.Resampling.LANCZOS)
        
        render = ImageTk.PhotoImage(img_display)
        self.panel.configure(image=render, text="")
        self.panel.image = render

        self.process_image(path)

    def process_image(self, path):
        # Preprocessing (Silent, no sliders)
        cv_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        h, w = cv_img.shape
        ratio = 32.0 / h
        new_w = int(w * ratio)
        img_resized = cv2.resize(cv_img, (new_w, 32))
        binary = cv2.adaptiveThreshold(img_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_pil = Image.fromarray(binary)

        # 1. Vision Inference
        transform = transforms.Compose([transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        image_tensor = transform(processed_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = self.vision_model(image_tensor)
            output = output.permute(1, 0, 2)
            raw_text = self.decode_vision(output)
            
        # 2. Neural NLP Inference
        corrected_text = self.run_nlp_correction(raw_text)
        
        # 3. Hybrid Verification (Safety Net)
        # If NLP result is not in DB, fallback to finding closest DB match
        if corrected_text not in MEDICINE_DB:
            matches = difflib.get_close_matches(corrected_text, MEDICINE_DB, n=1, cutoff=0.5)
            if matches:
                final_text = matches[0]
            else:
                # If NLP output was bad, try raw OCR against DB
                matches_raw = difflib.get_close_matches(raw_text, MEDICINE_DB, n=1, cutoff=0.5)
                final_text = matches_raw[0] if matches_raw else corrected_text
        else:
            final_text = corrected_text

        # Update UI
        self.lbl_raw.config(text=raw_text)
        self.lbl_nlp.config(text=final_text)
        
        if final_text in MEDICINE_DB:
            self.lbl_nlp.config(fg="#27AE60") # Green (Verified)
        else:
            self.lbl_nlp.config(fg="#E74C3C") # Red (Unknown)

        # Calculate Confidence
        confidence = difflib.SequenceMatcher(None, raw_text, final_text).ratio() * 100
        self.progress_var.set(confidence)
        self.lbl_conf_text.config(text=f"AI Consistency: {confidence:.1f}%")

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernApp(root)
    root.mainloop()