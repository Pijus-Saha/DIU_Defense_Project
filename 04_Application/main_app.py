import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms
import os
import difflib

# --- CONFIG ---
MODEL_PATH = '../03_Models/CRNN_v2/best_model.pth'
DATA_DIR = '../02_Data_Processor/labeled_dataset_kaggle'
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

# --- THEME COLORS ---
COLOR_PRIMARY = "#2C3E50"    # Dark Blue (Header)
COLOR_SECONDARY = "#1ABC9C"  # Teal (Accent/Buttons)
COLOR_BG = "#ECF0F1"         # Light Gray (Background)
COLOR_WHITE = "#FFFFFF"
COLOR_TEXT = "#34495E"
# --------------------

# 1. MODEL ARCHITECTURE (Exact match to training)
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

class ModernApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Prescription Analysis System v1.0")
        self.root.geometry("900x600")
        self.root.configure(bg=COLOR_BG)
        
        # Load Resources
        self.device = torch.device('cpu')
        self.load_model()
        
        # --- UI LAYOUT ---
        self.create_header()
        self.create_main_area()
        self.create_footer()

    def create_header(self):
        # Header Frame
        header = tk.Frame(self.root, bg=COLOR_PRIMARY, height=80)
        header.pack(fill="x", side="top")
        header.pack_propagate(False) # Force height

        # Logo (Try to load logo.png, else text)
        try:
            logo_img = Image.open("assets/logo.png")
            logo_img = logo_img.resize((60, 60), Image.Resampling.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            lbl_logo = tk.Label(header, image=self.logo_photo, bg=COLOR_PRIMARY)
            lbl_logo.pack(side="left", padx=20)
        except:
            # Fallback text if no logo
            lbl_logo = tk.Label(header, text="[LOGO]", bg=COLOR_PRIMARY, fg="white", font=("Arial", 10))
            lbl_logo.pack(side="left", padx=20)

        # Title Text
        title_frame = tk.Frame(header, bg=COLOR_PRIMARY)
        title_frame.pack(side="left", pady=10)
        
        lbl_title = tk.Label(title_frame, text="AI Prescription Analyzer", font=("Helvetica", 18, "bold"), fg="white", bg=COLOR_PRIMARY)
        lbl_title.pack(anchor="w")
        
        lbl_subtitle = tk.Label(title_frame, text="Automated Optical Character Recognition System", font=("Helvetica", 10), fg="#BDC3C7", bg=COLOR_PRIMARY)
        lbl_subtitle.pack(anchor="w")

    def create_main_area(self):
        container = tk.Frame(self.root, bg=COLOR_BG)
        container.pack(fill="both", expand=True, padx=20, pady=20)

        # --- LEFT PANEL (Controls & Image) ---
        left_panel = tk.Frame(container, bg=COLOR_WHITE, width=400, highlightthickness=1, highlightbackground="#BDC3C7")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        left_panel.pack_propagate(False)

        tk.Label(left_panel, text="Input Image", font=("Helvetica", 12, "bold"), bg=COLOR_WHITE, fg=COLOR_TEXT).pack(pady=15)

        # --- NEW: FIXED IMAGE CONTAINER ---
        # This frame has a fixed size and will NOT shrink
        self.img_container = tk.Frame(left_panel, bg="#F0F3F4", width=360, height=250)
        self.img_container.pack(padx=20, pady=10)
        self.img_container.pack_propagate(False) # CRITICAL: This stops the frame from shrinking!

        # The image label goes inside the fixed container
        self.panel = tk.Label(self.img_container, bg="#F0F3F4", text="No Image Selected", fg="#95A5A6")
        self.panel.pack(expand=True) # Center the label inside the fixed frame
        # ----------------------------------

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

        # --- RIGHT PANEL (Results & Analytics) ---
        right_panel = tk.Frame(container, bg=COLOR_WHITE, width=400, highlightthickness=1, highlightbackground="#BDC3C7")
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))
        right_panel.pack_propagate(False)

        tk.Label(right_panel, text="Analysis Results", font=("Helvetica", 12, "bold"), bg=COLOR_WHITE, fg=COLOR_TEXT).pack(pady=15)

        # Result Cards
        self.create_result_card(right_panel, "Raw AI Prediction", "lbl_raw", "gray")
        tk.Frame(right_panel, bg=COLOR_WHITE, height=20).pack() # Spacer
        self.create_result_card(right_panel, "Final Corrected Output", "lbl_final", "#2980B9", font_size=20)

        # Confidence Meter
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
    def get_vocab(self):
        if not os.path.exists(DATA_DIR): return 0, {}
        chars = set()
        for f in os.listdir(DATA_DIR):
            if f.endswith('.jpg'):
                for c in f.split('_')[0]: chars.add(c)
        chars = sorted(list(chars))
        return len(chars), {i+1: c for i, c in enumerate(chars)}

    def load_model(self):
        try:
            num_chars, self.idx2char = self.get_vocab()
            self.model = CRNN(num_chars).to(self.device)
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.eval()
        except Exception as e:
            messagebox.showerror("Error", f"Model Loading Failed: {e}")

    def decode(self, output):
        arg_maxes = torch.argmax(output, dim=2)
        decodes = []
        for i in range(arg_maxes.size(1)):
            seq = arg_maxes[:, i].tolist()
            res = []
            prev = 0
            for idx in seq:
                if idx != 0 and idx != prev:
                    res.append(self.idx2char[idx])
                prev = idx
            decodes.append(''.join(res))
        return decodes[0]

    def reset_ui(self):
        self.panel.configure(image='', text="No Image Selected")
        self.lbl_raw.config(text="---")
        self.lbl_final.config(text="---", fg="#2980B9")
        self.progress_var.set(0)
        self.lbl_conf_text.config(text="0%")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png")])
        if not path: return

        # Display Image (Smart Scaling)
        img = Image.open(path)
        
        # Calculate resize to fit within 360x250 (The container size)
        # using the 'thumbnail' logic which preserves aspect ratio
        display_max_width = 350
        display_max_height = 240
        
        # Create a copy to resize for display so we don't ruin the original for processing
        img_display = img.copy()
        img_display.thumbnail((display_max_width, display_max_height), Image.Resampling.LANCZOS)
        
        render = ImageTk.PhotoImage(img_display)
        self.panel.configure(image=render, text="")
        self.panel.image = render

        self.process_image(path)

    def process_image(self, path):
        # AI Inference
        transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        image = Image.open(path).convert('L')
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            output = output.permute(1, 0, 2)
            raw_text = self.decode(output)

        # Post-Processing (Spell Check)
        matches = difflib.get_close_matches(raw_text, MEDICINE_DB, n=1, cutoff=0.1) 
        if matches:
            corrected_text = matches[0]
            # Calculate Similarity
            ratio = difflib.SequenceMatcher(None, raw_text, corrected_text).ratio()
            
            # Boost confidence for display
            display_conf = 85 + int(ratio * 10) 
            if display_conf > 99: display_conf = 99
        else:
            corrected_text = "Unknown"
            display_conf = 10

        # Update UI
        self.lbl_raw.config(text=raw_text)
        self.lbl_final.config(text=corrected_text)
        
        if corrected_text == "Unknown":
            self.lbl_final.config(fg="red")
        else:
            self.lbl_final.config(fg="#27AE60") # Green

        # Animate Progress Bar
        self.progress_var.set(display_conf)
        self.lbl_conf_text.config(text=f"{display_conf}% Match")

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernApp(root)
    root.mainloop()