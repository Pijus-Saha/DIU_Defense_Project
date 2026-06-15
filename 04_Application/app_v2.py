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
# UPDATE THIS PATH TO YOUR ACTUAL MODEL FILE
VISION_MODEL_PATH = '../03_Models/EfficientNet_GRU/best_model.pth' 
DATA_DIR = '../02_Data_Processor/labeled_dataset_kaggle'
IMG_HEIGHT = 32
IMG_WIDTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- MODERN THEME PALETTE ---
COLORS = {
    "bg": "#1e1e2e",           # Dark background
    "card_bg": "#282a36",      # Slightly lighter card
    "accent": "#bd93f9",       # Purple accent
    "text_main": "#f8f8f2",    # White-ish text
    "text_sub": "#6272a4",     # Muted text
    "success": "#50fa7b",      # Green
    "danger": "#ff5555",       # Red
    "btn_text": "#282a36"      # Dark text for buttons
}

# --- MEDICINE DATABASE ---
MEDICINE_DB = [
    "Ace", "Aceta", "Alatrol", "Amodis", "Atrizin", "Axodin", "Az", "Azithrocin", 
    "Azyth", "Bacaid", "Backtone", "Baclofen", "Baclon", "Bacmax", "Beklo", 
    "Bicozin", "Canazole", "Candinil", "Cetisoft", "Conaz", "Dancel", "Denixil", 
    "Diflu", "Dinafex", "Disopan", "Esonix", "Esoral", "Etizin", "Exium", 
    "Fenadin", "Fexo", "Fexofast", "Filmet", "Fixal", "Flamyd", "Flexibac", 
    "Flexilax", "Flugal", "Ketocon", "Ketoral", "Ketotab", "Ketozol", "Leptic", 
    "Lucan-R", "Lumona", "M-Kast", "Maxima", "Maxpro", "Metro", "Metsina", 
    "Monas", "Montair", "Montene", "Montex", "Napa", "NapaExtend", "Nexcap", 
    "Nexum", "Nidazyl", "Nizoder", "Odmon", "Omastin", "Opton", "Progut", 
    "Provair", "Renova", "Rhinil", "Ritch", "Rivotril", "Romycin", "Rozith", 
    "Sergel", "Tamen", "Telfast", "Tridosil", "Trilock", "Vifas", "Zithrin", 
    "Algin", "Alphapress", "Arotml", "Artica", "B126", "Baemax", "Beltas", 
    "Bilastin", "Bispro", "Bukof", "Cefotil Plus", "Cinaron Plus", "Ciprin", 
    "Clavurox", "Comet", "Cortimax", "D-Cap", "Dermocin ointment", "Diapro MR", 
    "Domilux", "Doxicap", "Doxiva", "Ebatin", "Ebion", "Edeloss", "Erion Ointment", 
    "Famodin", "Filwel Gold", "Filwel Teen HM", "Finix", "Furoclav", "Gabarol-CR", 
    "Gastrum", "Gaviflux DX", "Hemofix FZ", "Indever", "Lingo", "Losarva", 
    "Lubilax", "MAxsulin", "Maxcoral DX", "Menaril", "Mirapro", "Napdas", 
    "Olmezest", "Ostocal", "Othera", "Oxat", "Perosa Cream", "Protinavit", 
    "Radex", "Remmo", "Rex", "Rocipro", "Rocovay", "Rolac", "Sedil", "Telmidip", 
    "Tenorix", "Trialon", "Tyclav", "Veracal", "Visral", "XPA XR", "Xelpro Mups", 
    "Xinc B", "Zolivox", "Zovia Silver", "ebatin", "traxcef"
]

# ==========================================
# 1. MODEL ARCHITECTURE
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
# 2. MODERN GUI CLASS
# ==========================================
class AdvancedMedicalOCR:
    def __init__(self, root):
        self.root = root
        self.root.title("FYP Medical OCR | Full Prescription Analysis")
        self.root.geometry("1100x800")
        self.root.configure(bg=COLORS["bg"])
        
        # State Variables
        self.model = None
        self.idx2char = None
        self.original_cv_image = None # The full high-res loaded image
        self.pil_image_original = None # Keep original PIL for high quality zoom
        self.base_scale = 1.0         # Initial scale to fit screen
        self.zoom_level = 1.0         # User zoom factor
        self.img_anchor_x = 0         # Where the image is placed on canvas
        self.img_anchor_y = 0
        
        # Cropping State
        self.rect_start_x = None
        self.rect_start_y = None
        self.rect_id = None
        
        # UI Setup
        self.setup_styles()
        self.build_ui()
        self.init_system()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Horizontal.TProgressbar", background=COLORS["accent"], troughcolor=COLORS["card_bg"], bordercolor=COLORS["bg"])
        self.root.option_add("*Background", COLORS["bg"])
        self.root.option_add("*Foreground", COLORS["text_main"])

    def build_ui(self):
        # --- HEADER ---
        header = tk.Frame(self.root, bg=COLORS["card_bg"], height=80, padx=20)
        header.pack(fill="x", side="top")
        
        tk.Label(header, text="Prescription AI", font=("Segoe UI", 24, "bold"), 
                 bg=COLORS["card_bg"], fg=COLORS["accent"]).pack(side="left")
        
        status_frame = tk.Frame(header, bg=COLORS["card_bg"])
        status_frame.pack(side="right")
        self.status_dot = tk.Label(status_frame, text="●", font=("Arial", 16), bg=COLORS["card_bg"], fg="red")
        self.status_dot.pack(side="left")
        self.status_lbl = tk.Label(status_frame, text="System Offline", bg=COLORS["card_bg"], fg=COLORS["text_sub"])
        self.status_lbl.pack(side="left", padx=5)

        # --- MAIN CONTENT GRID ---
        main = tk.Frame(self.root, bg=COLORS["bg"], padx=20, pady=20)
        main.pack(fill="both", expand=True)

        main.columnconfigure(0, weight=2) # Input (Large)
        main.columnconfigure(1, weight=0) # Arrow
        main.columnconfigure(2, weight=1) # Processed (Small)
        main.rowconfigure(0, weight=1)

        # 1. INPUT CARD (Interactive Canvas)
        self.input_frame = tk.Frame(main, bg=COLORS["card_bg"], padx=2, pady=2)
        self.input_frame.grid(row=0, column=0, sticky="nsew")
        
        # Instructions Label
        tk.Label(self.input_frame, text="1. Left-Click to Crop | Right-Click to Pan | Wheel to Zoom", 
                 bg=COLORS["card_bg"], fg=COLORS["text_sub"], font=("Segoe UI", 10, "bold")).pack(fill="x", pady=5)
        
        # Canvas Container for Scrollbars
        canvas_container = tk.Frame(self.input_frame, bg="black")
        canvas_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Scrollbars
        v_scroll = tk.Scrollbar(canvas_container, orient="vertical")
        h_scroll = tk.Scrollbar(canvas_container, orient="horizontal")
        
        # Canvas
        self.canvas_input = tk.Canvas(canvas_container, bg="#000000", cursor="cross",
                                      yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        # Configure Scrollbars
        v_scroll.config(command=self.canvas_input.yview)
        h_scroll.config(command=self.canvas_input.xview)
        
        # Layout
        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")
        self.canvas_input.pack(side="left", fill="both", expand=True)
        
        # Bind Mouse Events for Cropping
        self.canvas_input.bind("<ButtonPress-1>", self.on_crop_start)
        self.canvas_input.bind("<B1-Motion>", self.on_crop_drag)
        self.canvas_input.bind("<ButtonRelease-1>", self.on_crop_end)
        
        # Bind Mouse Events for Panning (Right Click)
        self.canvas_input.bind("<ButtonPress-3>", self.on_pan_start)
        self.canvas_input.bind("<B3-Motion>", self.on_pan_drag)
        
        # Bind Mouse Wheel Zoom
        self.canvas_input.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows
        self.canvas_input.bind("<Button-4>", lambda e: self.change_zoom(0.1)) # Linux Scroll Up
        self.canvas_input.bind("<Button-5>", lambda e: self.change_zoom(-0.1)) # Linux Scroll Down

        # Arrow
        tk.Label(main, text="➔", font=("Arial", 30), bg=COLORS["bg"], fg=COLORS["text_sub"]).grid(row=0, column=1, padx=20)

        # 2. PROCESSED CARD
        self.card_proc_frame = tk.Frame(main, bg=COLORS["card_bg"], padx=2, pady=2)
        self.card_proc_frame.grid(row=0, column=2, sticky="nsew")
        tk.Label(self.card_proc_frame, text="2. AI Vision (Cropped Region)", 
                 bg=COLORS["card_bg"], fg=COLORS["text_sub"], font=("Segoe UI", 10, "bold")).pack(fill="x", pady=5)
        
        # We use a Label here since it's just display
        self.lbl_proc_img = tk.Label(self.card_proc_frame, bg="#000000", text="Waiting for Crop...", fg="#444")
        self.lbl_proc_img.pack(fill="both", expand=True, padx=5, pady=5)

        # --- CONTROLS & RESULTS ---
        bottom_panel = tk.Frame(self.root, bg=COLORS["card_bg"], height=200, padx=40, pady=20)
        bottom_panel.pack(fill="x", side="bottom")

        # Control Buttons
        ctrl_frame = tk.Frame(bottom_panel, bg=COLORS["card_bg"])
        ctrl_frame.pack(side="left", pady=10)

        self.btn_load = tk.Button(ctrl_frame, text="UPLOAD PRESCRIPTION", command=self.load_image,
                                  bg=COLORS["accent"], fg=COLORS["btn_text"], font=("Segoe UI", 12, "bold"),
                                  relief="flat", padx=20, pady=10, cursor="hand2")
        self.btn_load.pack(side="left", padx=5)

        self.btn_reset = tk.Button(ctrl_frame, text="RESET", command=self.reset_ui,
                                  bg=COLORS["danger"], fg="white", font=("Segoe UI", 12, "bold"),
                                  relief="flat", padx=15, pady=10, cursor="hand2")
        self.btn_reset.pack(side="left", padx=5)

        # Zoom Controls
        self.btn_zoom_out = tk.Button(ctrl_frame, text="-", command=lambda: self.change_zoom(-0.2),
                                      bg="#34495e", fg="white", font=("Segoe UI", 14, "bold"),
                                      relief="flat", width=3, cursor="hand2")
        self.btn_zoom_out.pack(side="left", padx=5)
        
        self.btn_zoom_in = tk.Button(ctrl_frame, text="+", command=lambda: self.change_zoom(0.2),
                                     bg="#34495e", fg="white", font=("Segoe UI", 14, "bold"),
                                     relief="flat", width=3, cursor="hand2")
        self.btn_zoom_in.pack(side="left", padx=5)

        # Results Area
        res_frame = tk.Frame(bottom_panel, bg=COLORS["card_bg"])
        res_frame.pack(side="right", fill="both", expand=True, padx=(50, 0))

        self.create_result_row(res_frame, "Raw Model Output:", "lbl_raw", COLORS["text_sub"])
        self.create_result_row(res_frame, "Heuristic Safety Net:", "lbl_final", COLORS["success"], font_size=24)
        
        self.progress = ttk.Progressbar(res_frame, style="Horizontal.TProgressbar", length=300, mode='determinate')
        self.progress.pack(anchor="e", pady=(10, 5))
        self.lbl_conf = tk.Label(res_frame, text="Confidence: 0%", bg=COLORS["card_bg"], fg=COLORS["text_sub"])
        self.lbl_conf.pack(anchor="e")

    def create_result_row(self, parent, label_text, attr_name, color, font_size=12):
        f = tk.Frame(parent, bg=COLORS["card_bg"])
        f.pack(fill="x", pady=2)
        tk.Label(f, text=label_text, width=20, anchor="w", bg=COLORS["card_bg"], fg=COLORS["text_sub"]).pack(side="left")
        lbl = tk.Label(f, text="---", font=("Courier New", font_size, "bold"), bg=COLORS["card_bg"], fg=color)
        lbl.pack(side="left")
        setattr(self, attr_name, lbl)

    # ==========================================
    # LOGIC: INITIALIZATION
    # ==========================================
    def get_vision_vocab(self):
        try:
            chars = set()
            if not os.path.exists(DATA_DIR):
                return list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-'"), {}
            for f in os.listdir(DATA_DIR):
                if f.lower().endswith(('.jpg', '.png')):
                    label_part = f.split('_')[0]
                    for c in label_part: chars.add(c)
            chars = sorted(list(chars))
            return chars, {i+1: c for i, c in enumerate(chars)}
        except Exception as e:
            print(f"Vocab Error: {e}")
            return list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"), {}

    def init_system(self):
        try:
            self.vocab_chars, self.idx2char = self.get_vision_vocab()
            if os.path.exists(VISION_MODEL_PATH):
                self.model = EfficientNet_GRU(len(self.vocab_chars)).to(DEVICE)
                checkpoint = torch.load(VISION_MODEL_PATH, map_location=DEVICE)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                     self.model.load_state_dict(checkpoint['state_dict'])
                else:
                     try: self.model.load_state_dict(checkpoint)
                     except: print("Warning: Model mismatch")
                self.model.eval()
                self.status_dot.config(fg=COLORS["success"])
                self.status_lbl.config(text="System Online (Ready to Scan)")
            else:
                self.status_dot.config(fg="orange")
                self.status_lbl.config(text="Model Not Found (Demo UI Only)")
        except Exception as e:
            self.status_dot.config(fg="red")
            self.status_lbl.config(text="System Error")
            print(e)

    # ==========================================
    # LOGIC: CROP & IMAGE HANDLING
    # ==========================================
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if not path: return
        
        # 1. Load Full Res Image (OpenCV)
        self.original_cv_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if self.original_cv_image is None: return

        # 2. Store original PIL for resizing
        img_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        self.pil_image_original = Image.fromarray(img_rgb)
        
        # 3. Calculate Base Scale (Fit to Canvas)
        self.canvas_input.update() # Ensure dimensions are calculated
        c_width = self.canvas_input.winfo_width()
        c_height = self.canvas_input.winfo_height()
        if c_width < 10: c_width = 600 # Fallback
        if c_height < 10: c_height = 400
        
        w, h = self.pil_image_original.size
        scale_w = c_width / w
        scale_h = c_height / h
        self.base_scale = min(scale_w, scale_h, 1.0)
        self.zoom_level = 1.0
        
        self.redraw_canvas()

    def redraw_canvas(self):
        if self.pil_image_original is None: return
        
        # Calculate Target Size based on Zoom
        effective_scale = self.base_scale * self.zoom_level
        w, h = self.pil_image_original.size
        new_w = int(w * effective_scale)
        new_h = int(h * effective_scale)
        
        # Resize using High Quality Filter
        pil_resized = self.pil_image_original.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_input_img = ImageTk.PhotoImage(pil_resized)
        
        # Update Canvas
        self.canvas_input.delete("all")
        # Anchor at 0,0 for scrolling support
        self.canvas_input.create_image(0, 0, image=self.tk_input_img, anchor="nw")
        self.canvas_input.config(scrollregion=self.canvas_input.bbox("all"))

    def change_zoom(self, delta):
        if self.pil_image_original is None: return
        new_zoom = self.zoom_level + delta
        if 0.1 < new_zoom < 10.0: # Wider zoom limits
            self.zoom_level = new_zoom
            self.redraw_canvas()

    def on_mouse_wheel(self, event):
        # Windows mouse wheel
        if event.delta > 0:
            self.change_zoom(0.1)
        else:
            self.change_zoom(-0.1)

    def reset_ui(self):
        self.original_cv_image = None
        self.pil_image_original = None
        self.rect_id = None
        self.canvas_input.delete("all")
        self.lbl_proc_img.config(image="", text="Waiting for Crop...")
        if hasattr(self, 'tk_proc_img'): del self.tk_proc_img
        self.lbl_raw.config(text="---")
        self.lbl_final.config(text="---", fg=COLORS["success"])
        self.progress['value'] = 0
        self.lbl_conf.config(text="Confidence: 0%")

    # --- Mouse Events for Panning (Right Click) ---
    def on_pan_start(self, event):
        self.canvas_input.scan_mark(event.x, event.y)

    def on_pan_drag(self, event):
        self.canvas_input.scan_dragto(event.x, event.y, gain=1)

    # --- Mouse Events for Cropping (Left Click) ---
    def on_crop_start(self, event):
        if self.original_cv_image is None: return
        # Translate window coordinates to canvas coordinates
        self.rect_start_x = self.canvas_input.canvasx(event.x)
        self.rect_start_y = self.canvas_input.canvasy(event.y)
        
        if self.rect_id: self.canvas_input.delete(self.rect_id)
        self.rect_id = self.canvas_input.create_rectangle(
            self.rect_start_x, self.rect_start_y, self.rect_start_x, self.rect_start_y,
            outline=COLORS["success"], width=2, dash=(4, 2)
        )

    def on_crop_drag(self, event):
        if self.rect_id:
            cur_x = self.canvas_input.canvasx(event.x)
            cur_y = self.canvas_input.canvasy(event.y)
            self.canvas_input.coords(self.rect_id, self.rect_start_x, self.rect_start_y, cur_x, cur_y)

    def on_crop_end(self, event):
        if self.original_cv_image is None: return
        
        # 1. Get Canvas Coordinates (Account for Scroll)
        x1, y1 = self.rect_start_x, self.rect_start_y
        x2 = self.canvas_input.canvasx(event.x)
        y2 = self.canvas_input.canvasy(event.y)
        
        # Normalize
        x_start, x_end = sorted([x1, x2])
        y_start, y_end = sorted([y1, y2])
        
        # 2. Convert to Original Image Coordinates using Effective Scale
        # Since we anchor at 0,0, offset is 0. simple division.
        effective_scale = self.base_scale * self.zoom_level
        
        real_x1 = int(x_start / effective_scale)
        real_y1 = int(y_start / effective_scale)
        real_x2 = int(x_end / effective_scale)
        real_y2 = int(y_end / effective_scale)
        
        # Clamp bounds
        h, w = self.original_cv_image.shape
        real_x1 = max(0, real_x1); real_y1 = max(0, real_y1)
        real_x2 = min(w, real_x2); real_y2 = min(h, real_y2)
        
        # 3. Crop
        if (real_x2 - real_x1) > 5 and (real_y2 - real_y1) > 5:
            cropped_roi = self.original_cv_image[real_y1:real_y2, real_x1:real_x2]
            self.process_roi(cropped_roi)

    # ==========================================
    # LOGIC: PREPROCESSING & INFERENCE
    # ==========================================
    def preprocess_image(self, image_input):
        if isinstance(image_input, str):
            img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        else:
            img = image_input
        if img is None: return None, None

        img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
        binary = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        h, w = binary.shape
        ratio = 32.0 / h
        new_w = int(w * ratio)
        img_resized = cv2.resize(binary, (new_w, 32), interpolation=cv2.INTER_AREA)
        return img, img_resized

    def process_roi(self, roi_image):
        _, processed_bin = self.preprocess_image(roi_image)
        if processed_bin is None: return

        pil_bin = Image.fromarray(processed_bin)
        display_w = self.card_proc_frame.winfo_width()
        display_h = self.card_proc_frame.winfo_height()
        if display_w > 1:
             pil_display = pil_bin.resize((display_w, int(display_w * (32/processed_bin.shape[1]))), Image.Resampling.NEAREST)
             if pil_display.height > display_h:
                 pil_display = pil_bin.resize((int(display_h * (processed_bin.shape[1]/32)), display_h), Image.Resampling.NEAREST)
        else:
             pil_display = pil_bin.resize((200, 64), Image.Resampling.NEAREST)
             
        self.tk_proc_img = ImageTk.PhotoImage(pil_display)
        self.lbl_proc_img.config(image=self.tk_proc_img, text="")
        self.run_inference(processed_bin)

    def run_inference(self, processed_bin):
        if self.model is None:
            self.lbl_raw.config(text="[Demo Mode]")
            return

        pil_img = Image.fromarray(processed_bin)
        transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), 
            transforms.ToTensor(), 
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            output = output.permute(1, 0, 2)
            raw_text = self.decode_prediction(output)
            
        self.lbl_raw.config(text=raw_text)
        self.apply_safety_net(raw_text)

    def decode_prediction(self, output):
        arg_maxes = torch.argmax(output, dim=2)
        decodes = []
        for i in range(arg_maxes.size(1)):
            seq = arg_maxes[:, i].tolist()
            res = []
            prev = 0
            for idx in seq:
                if idx != 0 and idx != prev:
                    if idx in self.idx2char:
                        res.append(self.idx2char[idx])
                prev = idx
            decodes.append(''.join(res))
        return decodes[0] if decodes else ""

    def apply_safety_net(self, raw_text):
        if not raw_text: return
        matches = difflib.get_close_matches(raw_text, MEDICINE_DB, n=1, cutoff=0.4)
        if matches:
            final_text = matches[0]
            similarity = difflib.SequenceMatcher(None, raw_text, final_text).ratio()
            conf = int(similarity * 100)
            self.lbl_final.config(text=final_text, fg=COLORS["success"])
            self.progress['value'] = conf
            self.lbl_conf.config(text=f"Confidence: {conf}% (Verified)")
        else:
            self.lbl_final.config(text="Unknown", fg=COLORS["danger"])
            self.progress['value'] = 20
            self.lbl_conf.config(text="Confidence: Low (No Match)")

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedMedicalOCR(root)
    root.mainloop()