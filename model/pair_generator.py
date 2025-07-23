import itertools
import random
import cv2
import numpy as np
from pathlib import Path

import tqdm
from config import IMAGE_SIZE
import albumentations as A
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import sys
import os

from model.coinclip import get_coinclip_embeddings
from utils.visualize import analyze_similarity_distribution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_all_images_from_group(group_path):
    images = []
    for img_path in Path(group_path).glob("*.jpg"):
        images.append(img_path)
    return images

def augment_image(img):
    transform = A.Compose([
        A.RandomCrop(height=80, width=80, p=0.5),
        A.Resize(height=100, width=100),  # Wiederherstellen der Zielgröße
        A.ColorJitter(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Rotate(limit=20, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.RandomShadow(p=0.2),
        A.Affine(p=0.5, scale=(0.8, 1.2), translate_percent=(0.1, 0.1)),
    ])
    return transform(image=img)["image"]

def make_self_supervised_pairs(images, n_pairs=1000):
    pairs = []
    labels = []

    #images = [cv2.resize(cv2.imread(str(p)), IMAGE_SIZE) for p in images if cv2.imread(str(p)) is not None]
    debug_dir = "debug/black_images"
    loaded_images = []
    for p in images:
        img = load_and_check_image(p, size=IMAGE_SIZE, debug_dir=debug_dir)
        if img is not None:
            loaded_images.append(img)
    images = loaded_images
    
    for _ in range(n_pairs):
        # Positive Pair (gleiches Bild, augmentiert)
        img = random.choice(images)
        img_aug1 = augment_image(img)
        img_aug2 = augment_image(img)
        pairs.append([img_aug1, img_aug2])
        labels.append(1)

        # Negative Pair (zwei verschiedene Bilder)
        img1, img2 = random.sample(images, 2)
        pairs.append([img1, img2])
        labels.append(0)
    
    return np.array(pairs), np.array(labels)

def load_and_check_image(img_path, size=IMAGE_SIZE, debug_dir=None):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Konnte Bild nicht laden: {img_path}")
        return None
    img = cv2.resize(img, size)
    img = img.astype("float32") / 255.0

    if np.all(img == 0):
        print(f"⚠️ Bild komplett schwarz: {img_path}")
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(str(Path(debug_dir) / f"black_{Path(img_path).name}"), img * 255)
        return None

    return img

def make_similarity_pairs(model, image_paths, mypath, upper_threshold=0.75, lower_threshold=0.90, coin_side='reverse', group=None):
    # Lade Bilder & normalisiere
    #imgs = [cv2.resize(cv2.imread(str(p)), IMAGE_SIZE) for p in image_paths if cv2.imread(str(p)) is not None]
    debug_dir = "debug/black_images"
    loaded_images = []
    for p in image_paths:
        img = load_and_check_image(p, size=IMAGE_SIZE, debug_dir=debug_dir)
        if img is not None:
            loaded_images.append(img)
    imgs = np.array(loaded_images)
    print(f"Anzahl geladener Bilder in make_similarity_pairs: {len(imgs)}   and {len(image_paths)}")

    embedding_model = model.get_layer("embedding_model")
    emb_self = embedding_model.predict(imgs, verbose=0)
    
    # Get embeddings from CoinClip
    emb_coinclip = get_coinclip_embeddings(imgs, coin_side, group)

    # Concatenate both
    embedding_combined = combine_embeddings(emb_self, emb_coinclip, method="weighted_concat", alpha=0.6, normalize=True)

    sim_matrix = cosine_similarity(embedding_combined)
    analyze_similarity_distribution(sim_matrix, out_path=mypath, group=group, model_type='siamese', coin_side=coin_side)

    pairs = []
    labels = []
    n = len(image_paths)

    # for i in range(n):
    #     for j in range(i + 1, n):
    #         sim = sim_matrix[i, j]
    #         if sim > upper_threshold:
    #             pairs.append((imgs[i], imgs[j]))
    #             labels.append(1)
    #         elif sim < lower_threshold:
    #             pairs.append((imgs[i], imgs[j]))
    #             labels.append(0)
    max_pairs_per_label = 5000
    positive_pairs = []
    negative_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            if sim > upper_threshold and len(positive_pairs) < max_pairs_per_label:
                positive_pairs.append((imgs[i], imgs[j]))
            elif sim < lower_threshold and len(negative_pairs) < max_pairs_per_label:
                negative_pairs.append((imgs[i], imgs[j]))

    pairs = positive_pairs + negative_pairs
    labels = [1]*len(positive_pairs) + [0]*len(negative_pairs)
    if len(set(labels)) < 2:
        print("⚠️ WARNUNG: Nur ein Label-Typ generiert. Prüfe Threshold oder Datenvielfalt.")
    
    print(f"✅ Ähnliche Paare (label=1): {labels.count(1)}")
    print(f"✅ Ungleiche Paare (label=0): {labels.count(0)}")
    
    pairs = np.array(pairs)
    labels = np.array(labels)
    return pairs, labels


def combine_embeddings(emb_self, emb_coinclip, method="concat", alpha=0.5, normalize=True):
    if normalize:
        from sklearn.preprocessing import normalize
        emb_self = normalize(emb_self)
        emb_coinclip = normalize(emb_coinclip)

    if method == "concat":
        return np.concatenate([emb_self, emb_coinclip], axis=1)
    elif method == "weighted_concat":
        return np.concatenate([alpha * emb_self, (1 - alpha) * emb_coinclip], axis=1)
    elif method == "add":
        return alpha * emb_self + (1 - alpha) * emb_coinclip
    else:
        raise ValueError("Unknown combination method")
