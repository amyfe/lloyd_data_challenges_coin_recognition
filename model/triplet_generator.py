# model/triplet_generator.py

import random
import cv2
import numpy as np
from config import IMAGE_SIZE
from pathlib import Path
from model.coinclip import get_coinclip_embeddings
from model.pair_generator import augment_image, combine_embeddings, load_and_check_image
from sklearn.metrics.pairwise import cosine_similarity

from utils.visualize import alpha_sweep_pca, analyze_similarity_distribution

def make_self_supervised_triplets(image_paths, n_triplets=5000, embedding_model=None):
    """
    Erzeugt Triplets mit:
    - Anchor: Originalbild
    - Positive: augmentierte Variante desselben Bilds
    - Negative: augmentierte Variante eines anderen Bilds
    """
    # Lade und resize Bilder
    loaded_images = []
    debug_dir = "debug/black_images"
    for p in image_paths:
        img = load_and_check_image(p, size=IMAGE_SIZE, debug_dir=debug_dir)
        if img is not None:
            loaded_images.append(img)
    images = images = np.array(loaded_images)  
    if embedding_model:
        print("📐 Berechne Embeddings für Triplet-Auswahl...")
        embeddings = embedding_model.predict(images, verbose=0)
        return mine_triplets_from_embeddings(images, embeddings)
    triplets = []
    for _ in range(n_triplets):
        anchor = random.choice(images)
        positive = augment_image(anchor)
        # Negative aus anderem Bild
        while True:
            negative = random.choice(images)
            if not np.array_equal(negative, anchor):
                break
        negative = augment_image(negative)

        triplets.append((anchor, positive, negative))

    anchors, positives, negatives = zip(*triplets)
    return np.array(anchors), np.array(positives), np.array(negatives)

def make_triplets_from_model(model, image_paths, my_path, group, coin_side="reverse", upper_threshold=0.75, lower_threshold=0.9, max_triplets=5000, alpha=0.4):
    """
    Erzeugt Triplets mit einem Modell basierend auf Ähnlichkeit im Embedding-Raum:
    - Anchor & Positive: hohe Cosine Similarity (> upper_threshold)
    - Anchor & Negative: niedrige Similarity (< lower_threshold)
    """
    print("📐 Berechne Embeddings für Triplet-Auswahl...")

    # Bilder laden & normalisieren
    imgs = []
    valid_paths = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, IMAGE_SIZE)
        imgs.append(img.astype("float32") / 255.0)
        valid_paths.append(path)

    imgs = np.array(imgs)
    n = len(imgs)

    embedding_model = model.get_layer("Embedding")
    emb_self = embedding_model.predict(imgs, verbose=0)
    
    # Get embeddings from CoinClip
    emb_coinclip = get_coinclip_embeddings(imgs, coin_side, group)
    # Concatenate both
    embedding_combined = combine_embeddings(emb_self, emb_coinclip, method="weighted_concat", alpha=alpha, normalize=True)

    sim_matrix = cosine_similarity(embedding_combined)
    print(f"Image Paths: {my_path}")
    analyze_similarity_distribution(sim_matrix, out_path=my_path, group=group, model_type='triplet', coin_side=coin_side)
    
    triplets = []
    count = 0

    for i in range(n):
        pos_indices = np.where(sim_matrix[i] > upper_threshold)[0]
        pos_indices = [j for j in pos_indices if j != i]
        if not pos_indices:
            continue

        neg_indices = np.where(sim_matrix[i] < lower_threshold)[0]
        if not neg_indices.any():
            continue

        pos_idx = random.choice(pos_indices)
        neg_idx = random.choice(neg_indices)

        anchor = imgs[i]
        positive = imgs[pos_idx]
        negative = imgs[neg_idx]

        triplets.append((anchor, positive, negative))
        count += 1

        if count >= max_triplets:
            break

    if not triplets:
        raise ValueError("Keine Triplets erzeugt. Thresholds zu streng?")

    anchors, positives, negatives = zip(*triplets)
    return np.stack(anchors), np.stack(positives), np.stack(negatives)

def mine_triplets_from_embeddings(images, embeddings, upper=0.8, lower=0.3, max_triplets=5000):
    sim_matrix = cosine_similarity(embeddings)
    n = len(images)
    triplets = []
    for i in range(n):
        anchor = images[i]
        sims = sim_matrix[i]
        # Positives = ähnliche, aber nicht identische
        pos_idx = np.where((sims > upper) & (np.arange(n) != i))[0]
        # Negatives = unähnliche
        neg_idx = np.where(sims < lower)[0]

        for p in pos_idx:
            for n_ in neg_idx:
                if len(triplets) >= max_triplets:
                    break
                positive = images[p]
                negative = images[n_]
                triplets.append((anchor, positive, negative))
    if not triplets:
        raise ValueError("⚠️ Keine Triplets gefunden – prüfe Thresholds.")
    anchors, positives, negatives = zip(*triplets)
    return np.array(anchors), np.array(positives), np.array(negatives)
