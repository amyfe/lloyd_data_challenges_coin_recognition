import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from config import IMAGE_SIZE, TRIPLET_MODEL_PATH


def load_and_preprocess_images(image_paths):
    images = []
    valid_paths = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is not None:
            img = cv2.resize(img, IMAGE_SIZE)
            img = img.astype("float32") / 255.0
            images.append(img)
            valid_paths.append(path)
    return np.array(images), valid_paths


def compute_similarity_matrix(embedding_model, images):
    embeddings = embedding_model.predict(images, verbose=0)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim_matrix = cosine_similarity(embeddings)
    return sim_matrix, embeddings


def visualize_similarity_matrix(sim_matrix, paths, group):
    labels = [p.name[:12] for p in paths]  # Kürzere Namen
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_matrix, xticklabels=labels, yticklabels=labels, cmap="viridis", square=True)
    plt.title(f"Similarity-Matrix Gruppe {group}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def cluster_embeddings(embeddings, paths, group, distance_threshold=0.2):
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage="average")
    labels = clustering.fit_predict(embeddings)
    for cluster_id in sorted(set(labels)):
        print(f"\n🧩 Cluster {cluster_id}:")
        cluster_members = [p.name for i, p in enumerate(paths) if labels[i] == cluster_id]
        for name in cluster_members:
            print(f"  - {name}")


def test_triplet_model(group: str, coin_side: str, image_paths: list):
    print(f"🔍 Lade Triplet-Modell für Gruppe {group} ({coin_side})")
    model_path = TRIPLET_MODEL_PATH / f"{group}_{coin_side}_triplet_final.keras"
    model = load_model(model_path)

    embedding_model = model.get_layer("Embedding")

    print("📸 Lade und verarbeite Bilder...")
    images, valid_paths = load_and_preprocess_images(image_paths)

    print("📊 Berechne Ähnlichkeitsmatrix...")
    sim_matrix, embeddings = compute_similarity_matrix(embedding_model, images)

    visualize_similarity_matrix(sim_matrix, valid_paths, group)

    print("🔗 Starte Clustering basierend auf Embeddings...")
    cluster_embeddings(embeddings, valid_paths, group)


# Beispielnutzung
# test_triplet_model("A", image_paths, "reverse")
