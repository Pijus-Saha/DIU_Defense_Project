import cv2
import os
import uuid

# --- CONFIGURATION ---
SOURCE_FOLDER = 'raw_images' 
DEST_FOLDER = '../02_Data_Processor/labeled_dataset'

# I changed this from 600 to 900. 
# It should be big enough to read, but fit on your Asus screen.
DISPLAY_HEIGHT = 900 
# ---------------------

def resize_for_display(image, target_height):
    (h, w) = image.shape[:2]
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    return cv2.resize(image, (new_width, target_height))

def main():
    if not os.path.exists(SOURCE_FOLDER):
        os.makedirs(SOURCE_FOLDER)
        print(f"Created '{SOURCE_FOLDER}'. Add images and run again.")
        return

    if not os.path.exists(DEST_FOLDER):
        os.makedirs(DEST_FOLDER)

    images = [f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print(f"No images found in '{SOURCE_FOLDER}'!")
        return

    print(f"Found {len(images)} images.")
    print("--- CONTROLS ---")
    print("1. Draw Box -> Press SPACEBAR")
    print("2. Type Name -> Press ENTER")
    print("3. Press 'c' to skip image")

    for img_name in images:
        img_path = os.path.join(SOURCE_FOLDER, img_name)
        original_img = cv2.imread(img_path)
        
        if original_img is None: continue

        # Resize specifically for your screen
        display_img = resize_for_display(original_img, DISPLAY_HEIGHT)
        scale_factor = original_img.shape[0] / DISPLAY_HEIGHT

        while True:
            try:
                # Window opens here
                roi = cv2.selectROI("CROPPER", display_img, showCrosshair=True)
            except:
                break

            # If 'c' is pressed or window closed
            if roi[2] == 0 or roi[3] == 0:
                cv2.destroyAllWindows()
                break

            # Calculate real coordinates
            x = int(roi[0] * scale_factor)
            y = int(roi[1] * scale_factor)
            w = int(roi[2] * scale_factor)
            h = int(roi[3] * scale_factor)

            if w > 0 and h > 0:
                im_crop = original_img[y:y+h, x:x+w]
                
                # Show small preview
                cv2.imshow("Preview", im_crop)
                cv2.waitKey(100)

                label = input(f"Label for crop: ").strip()
                
                if label:
                    # Save file
                    unique_id = str(uuid.uuid4())[:8]
                    filename = f"{label}_{unique_id}.jpg"
                    save_path = os.path.join(DEST_FOLDER, filename)
                    cv2.imwrite(save_path, im_crop)
                    print(f"   Saved -> {filename}")
                
                cv2.destroyWindow("Preview")

    print("All done!")

if __name__ == "__main__":
    main()