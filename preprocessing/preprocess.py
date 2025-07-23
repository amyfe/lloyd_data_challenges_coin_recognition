import os
import cv2
import numpy as np
from pathlib import Path
from skimage.filters import threshold_otsu
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import IMAGE_SIZE, DATA_DIR, PREPROCESSED_DIR

def center_and_crop(image, mask):
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = image[y:y+h, x:x+w]
    max_dim = max(w, h)
    canvas = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    x_offset = (max_dim - w) // 2
    y_offset = (max_dim - h) // 2
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    return canvas

def preprocess_image(path):
    img = cv2.imread(str(path))
    if img is None:
        print(f"[WARN] Bild konnte nicht geladen werden: {path}")
        return None
    img, mask = remove_background(img)
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.medianBlur(img, 5)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    return img

def remove_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold for white (tune value as needed)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_3ch = cv2.merge([mask, mask, mask])
    result = cv2.bitwise_and(image, mask_3ch)
    return result, mask

def batch_preprocess_folder(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for class_folder in input_dir.iterdir():
        if class_folder.is_dir():
            out_class_folder = output_dir / class_folder.name
            out_class_folder.mkdir(parents=True, exist_ok=True)

            for image_file in class_folder.glob("*.jpg"):
                processed = preprocess_image(image_file)
                if processed is not None:
                    save_path = out_class_folder / image_file.name
                    cv2.imwrite(str(save_path), processed)

if __name__ == "__main__":
    print("🚀 Starte Preprocessing...")
    batch_preprocess_folder(DATA_DIR, PREPROCESSED_DIR)
    print("✅ Preprocessing abgeschlossen.")
