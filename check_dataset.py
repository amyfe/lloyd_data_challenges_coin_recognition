from pathlib import Path
import cv2


def check_preprocessed_images(path):
    base_dir = Path(path)
    total = 0
    for cls_folder in sorted(base_dir.glob("*")):
        if cls_folder.is_dir():
            images = list(cls_folder.glob("*.jpg"))
            print(f"{cls_folder.name}: {len(images)} Bilder")
            total += len(images)
    print(f"➡️ Insgesamt {total} Bilder gefunden.")

def check_minimum_images_per_class(path, min_images=2):
    base_dir = Path(path)
    for cls_folder in sorted(base_dir.glob("*")):
        if cls_folder.is_dir():
            count = len(list(cls_folder.glob("*.jpg")))
            if count < min_images:
                print(f"⚠️ Klasse {cls_folder.name} hat nur {count} Bilder!")



def check_image_sizes(path):
    base_dir = Path(path)
    for img_path in base_dir.glob("*/*.jpg"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"🚫 Fehler beim Laden: {img_path}")
            continue
        h, w = img.shape[:2]
        if (h, w) != (100, 100):  
            print(f"⚠️ {img_path} hat Größe {w}x{h}")
