import matplotlib.pyplot as plt
import sys
import os

from sklearn.metrics.pairwise import cosine_similarity
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
from config import IMAGE_SIZE
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import pandas as pd

def show_pair(img1, img2, label, pred=None):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img1.astype('float32'))
    axs[0].axis("off")
    axs[1].imshow(img2.astype('float32'))
    axs[1].axis("off")
    title = f"Label: {label}"
    if pred is not None:
        title += f" | Prediction: {pred:.2f}"
    plt.suptitle(title)

def generate_similarity_matrix(embedding_model, images, path, group, model_type = 'siamese', coin_side="reverse"):
    from scipy.spatial.distance import pdist, squareform
    from sklearn.metrics.pairwise import cosine_similarity

    imgs = [cv2.resize(cv2.imread(str(p)), IMAGE_SIZE) for p in images]
    imgs = np.array(imgs).astype("float32") / 255.0

    embeddings = embedding_model.predict(imgs)
    sim_matrix = cosine_similarity(embeddings)

    # Optional: Save the similarity matrix as a CSV
    os.makedirs("embeddings_result", exist_ok=True)
    # Full path to save the CSV
    file_path = os.path.join("embeddings_result", f'embeddings_similarity_matrix_{model_type}_{coin_side}_{group}.csv')
    np.savetxt(file_path, sim_matrix, delimiter=',')
    print("🐥🐥Typen der Similarity-Matrix:", type(sim_matrix))
    print("🐥🐥Similarity-Matrix Form:", sim_matrix.shape)

    plt.figure(figsize=(10, 8))
    plt.imshow(sim_matrix, cmap="viridis")
    plt.colorbar()
    plt.title(f"Ähnlichkeitsmatrix Gruppe {group}")
    matrix_dir = os.path.join(path, "visuals", f"{model_type}_models", "similarity_matrix")
    os.makedirs(matrix_dir, exist_ok=True)
    plt.savefig(os.path.join(matrix_dir, f"sim_matrix_{coin_side}_{group}.png"))
    plt.close()

import seaborn as sns

def analyze_similarity_distribution(sim_matrix, out_path=None, group="X", model_type='siamese', coin_side="reverse", extra=""):

    sims = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]  # nur obere Dreieckswerte
    print(f"📊 Cosine Similarity – Statistik:")
    print(f"🔹 Min: {sims.min():.4f}")
    print(f"🔹 Max: {sims.max():.4f}")
    print(f"🔹 Mittelwert: {sims.mean():.4f}")
    print(f"🔹 Median: {np.median(sims):.4f}")
    print(f"🔹 25. Percentil: {np.percentile(sims, 25):.4f}")
    print(f"🔹 75. Percentil: {np.percentile(sims, 75):.4f}")

    # Histogramm
    plt.figure(figsize=(7, 4))
    sns.histplot(sims, bins=30, kde=True, color="steelblue")
    plt.title(f"Verteilung der Cosine Similarities – Gruppe {group}")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Anzahl Paare")

    sim_dir = os.path.join(out_path, "visuals", f"{model_type}_models", "similarity_distribution")
    os.makedirs(sim_dir, exist_ok=True)
    plt.savefig(os.path.join(sim_dir, extra, f"cosine_distribution-{coin_side}_group_{group}.png"))
    plt.close()

def alpha_sweep_pca(emb_self, emb_coinclip, alphas=np.linspace(0.0, 1.0, 11), normalize_input=True):
    #zB aufrufen durch alpha_sweep_pca(emb_self, emb_coinclip, alphas=np.linspace(0.0, 1.0, 11), normalize_input=True)
    if normalize_input:
        emb_self = normalize(emb_self)
        emb_coinclip = normalize(emb_coinclip)

     # Passe die Dimension von emb_coinclip per PCA an emb_self an
    target_dim = emb_self.shape[1]
    pca = PCA(n_components=target_dim)
    emb_coinclip_reduced = pca.fit_transform(emb_coinclip)

    # Metriken vorbereiten
    avg_cosine_scores = []

    for alpha in alphas:
        emb_comb = alpha * emb_self + (1 - alpha) * emb_coinclip_reduced
        emb_comb = normalize(emb_comb)  # Sicherstellen, dass Cosine sinnvoll ist

        sim_matrix = cosine_similarity(emb_comb)
        upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        mean_sim = np.mean(upper_triangle)

        avg_cosine_scores.append(mean_sim)
        print(f"Alpha={alpha:.2f} → Ø Cosine Similarity: {mean_sim:.4f}")
    plt.figure(figsize=(6, 4))
    plt.plot(alphas, avg_cosine_scores, marker='o')
    plt.title("Durchschnittliche Cosine Similarity vs Alpha")
    plt.xlabel("Alpha (Gewicht auf self-supervised)")
    plt.ylabel("Ø Cosine Similarity")
    plt.grid(True)
    plt.tight_layout()
    now= pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    sim_dir = os.path.join("visuals", f"triplet_models", "alpha_distribution")
    os.makedirs(sim_dir, exist_ok=True)
    plt.savefig(os.path.join(sim_dir, f"alphas_{now}.png"))
    plt.close()
    #plt.show()

def check_embedding_consistency(combined_emb, labels=None):    
    pca = PCA(n_components=128)
    pca.fit(combined_emb)
    print(pca.explained_variance_ratio_)
    print("Summe:", np.sum(pca.explained_variance_ratio_))
    reduced = pca.fit_transform(combined_emb)
    # Zugriff auf die PCA-Komponenten (Form: [n_components, embedding_dim])
    components = pca.components_

    # Für Komponente 0 (die wichtigste):
    weights = components[0]  # Das sind die Gewichte für jede Embedding-Dimension (z. B. 128 Werte)

    # Sortiere die größten Beiträge:
    top_indices = np.argsort(-np.abs(weights))  # Nach Betrag sortieren
    print("Top Embedding-Dimensionen für PCA-Komponente 1:")
    for i in top_indices[:10]:
        print(f"Dimension {i}: Gewicht = {weights[i]:.4f}")

    plt.figure(figsize=(8, 6))
    if labels is not None:
        plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='coolwarm', s=12)
        plt.colorbar(label="Label")
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], s=10, alpha=0.6)
    plt.title("PCA der kombinierten Embeddings")
    plt.xlabel("PCA Komponente 1")
    plt.ylabel("PCA Komponente 2")
    plt.grid(True)
    plt.show()

def visualize_predictions(model, X1_val, X2_val, y_val, path, group=None, model_type = 'siamese', coin_side="reverse"):
    # Confusion Matrix
    y_pred = model.predict([X1_val, X2_val]) > 0.5
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Nicht gleich", "Gleich"])
    disp.plot(cmap="Blues")
    matrix_dir = os.path.join(path, "visuals", f"{model_type}_models", "confusion_matrix")
    os.makedirs(matrix_dir, exist_ok=True)

    if group and coin_side:
        matrix_file = os.path.join(matrix_dir, f"confusion_matrix_{coin_side}_group_{group}.png")
        matrix_file_html = os.path.join(matrix_dir, f"confusion_matrix_{coin_side}_group_{group}.html")

    elif group:
        matrix_file = os.path.join(matrix_dir, f"confusion_matrix_group_{group}.png")
        matrix_file_html = os.path.join(matrix_dir, f"confusion_matrix_group_{group}.html")
    else:
        matrix_file = os.path.join(matrix_dir, "confusion_matrix.png")
        matrix_file_html = os.path.join(matrix_dir, "confusion_matrix.html")

    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Nicht gleich", "Gleich"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (Validierung)")
    plt.savefig(matrix_file)
    plt.close()

    # t-SNE-Plot
    print("Berechne t-SNE Embeddings:")
    # Verwende nur eine Seite (z. B. X1) und das Embedding-Modell
    embedding_model = model.get_layer("embedding_model")
    embeddings = embedding_model.predict(X1_val)

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # t-SNE Plot
    tsne_dir = os.path.join(path, "visuals", f"{model_type}_models", "t_sne")
    os.makedirs(tsne_dir, exist_ok=True)
    tsne_file = os.path.join(tsne_dir, f"confusion_matrix_group_{coin_side}_{group}.png" if group else "confusion_matrix.png")
    tsne_file_html = os.path.join(tsne_dir, f"confusion_matrix_group_{coin_side}_{group}.html" if group else "confusion_matrix.html")

    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y_val, cmap="coolwarm", s=10)
    plt.title("t-SNE der Embeddings (X1, validierung)")
    plt.colorbar(label="Label (0 ≠, 1 =)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.savefig(tsne_file)
    plt.close()

    # Save as interactive HTML using plotly
    try:
        import plotly.express as px
        df = pd.DataFrame(embeddings_2d, columns=["TSNE-1", "TSNE-2"])
        df["Label"] = y_val
        fig = px.scatter(
            df, x="TSNE-1", y="TSNE-2", color="Label",
            title="t-SNE der Embeddings (X1, validierung)",
            color_continuous_scale="RdBu"   
        )
        fig.write_html(tsne_file_html)
    except ImportError:
        print("Plotly is not installed. Skipping HTML export.")

def save_prediction_pairs(model, X1_val, X2_val, y_val, path, group="X", model_type = 'siamese', coin_side="reverse"):
    print("🔍 Speichere Vorhersagepaare als Bilder...")
    y_pred = model.predict([X1_val, X2_val])
    dir_pairs = os.path.join(path, "visuals", f"{model_type}_models", "prediction_pairs")
    os.makedirs(dir_pairs, exist_ok=True)
    for i, pred in enumerate(y_pred):
        true_label = "gleich" if y_val[i] == 1 else "ungleich"
        pred_label = "gleich" if pred > 0.5 else "ungleich"
        img1 = (X1_val[i] * 255).astype(np.uint8)
        img2 = (X2_val[i] * 255).astype(np.uint8)
        if np.all(img1 == 0) or np.all(img2 == 0):
            print(f"⚠️ Warning: Image {i} is completely black.")
            continue
        concat = np.hstack((img1, img2))
        filename = f"{i:04d}_true_{true_label}_pred_{pred_label}.jpg"
        filepath = os.path.join(dir_pairs, group, f"{coin_side}_group_{group}_{filename}")
        cv2.imwrite(filepath, concat)

def save_similarity_matrix(sim_matrix, out_path, group="X", model_type='siamese', coin_side="reverse"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap="viridis")
    plt.title(f"Trainmodell Ähnlichkeitsmatrix – Gruppe {group}")
    sim_dir = os.path.join(out_path, "visuals", f"{model_type}_models", "similarity_matrix")
    os.makedirs(sim_dir, exist_ok=True)
    plt.savefig(os.path.join(sim_dir, f"train_similarity_matrix_{coin_side}_group_{group}.png"))
    plt.close()

## Triplet-Specific Functions
def visualize_triplet_embeddings(embedding_model, X_val, pos_val, neg_val, path=None, group="X", coin_side="reverse"):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    emb_anchor = embedding_model.predict(X_val)
    emb_pos = embedding_model.predict(pos_val)
    emb_neg = embedding_model.predict(neg_val)

    # Optional: Mittelwert oder nur Anchor
    combined = np.concatenate([emb_anchor, emb_pos, emb_neg], axis=0)
    labels = np.concatenate([
        np.zeros(len(emb_anchor)),         # 0 = anchor
        np.ones(len(emb_pos)),             # 1 = positive
        np.full(len(emb_neg), 2)           # 2 = negative
    ])

    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)
    tsne = TSNE(n_components=2, perplexity=min(30, len(combined) // 5), random_state=42)
    reduced = tsne.fit_transform(combined_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="coolwarm", s=10)
    plt.title("t-SNE der Triplet-Embeddings (Anchor vs Positive vs Negative)")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.grid(True)
    plt.legend(handles=scatter.legend_elements()[0], labels=["Anchor", "Positive", "Negative"])

    tsne_dir = os.path.join(path, "visuals", "triplet_models", "t_sne")
    os.makedirs(tsne_dir, exist_ok=True)
    plt.savefig(os.path.join(tsne_dir, f"tsne_{coin_side}_group_{group}.png"))
    plt.close()

def save_triplet_prediction_pairs(embedding_model, X_anchor, X_positive, X_negative, path, group="X", coin_side="reverse", extra=""):
    print("🔍 Speichere Triplet-Paare als Bilder...")

    emb_anchor = embedding_model.predict(X_anchor)
    emb_positive = embedding_model.predict(X_positive)
    emb_negative = embedding_model.predict(X_negative)

    dir_pairs = os.path.join(path, "visuals", "triplet_models", "prediction_pairs")
    os.makedirs(dir_pairs, exist_ok=True)
    print(f"Anzahl Triplet-Paare: {len(emb_anchor)} und {len(emb_positive)} und {len(emb_negative)}")
    # Save anchor-positive pairs
    for i in range(len(emb_anchor)):
        sim_pos = cosine_similarity(emb_anchor[i].reshape(1, -1), emb_positive[i].reshape(1, -1))[0, 0]
        img_anchor = (X_anchor[i] * 255).astype(np.uint8)
        img_positive = (X_positive[i] * 255).astype(np.uint8)
        concat_pos = np.hstack((img_anchor, img_positive))
        filename_pos = f"group_{group}_anchor_positive_{i:04d}_sim_{sim_pos:.2f}.jpg"
        filepath_pos = os.path.join(dir_pairs, coin_side, extra, filename_pos)
        cv2.imwrite(filepath_pos, concat_pos)

    # Save anchor-negative pairs
    for i in range(len(emb_anchor)):
        sim_neg = cosine_similarity(emb_anchor[i].reshape(1, -1), emb_negative[i].reshape(1, -1))[0, 0]
        img_anchor = (X_anchor[i] * 255).astype(np.uint8)
        img_negative = (X_negative[i] * 255).astype(np.uint8)
        concat_neg = np.hstack((img_anchor, img_negative))
        filename_neg = f"group_{group}_anchor_negative_{i:04d}_sim_{sim_neg:.2f}.jpg"
        filepath_neg = os.path.join(dir_pairs,coin_side, filename_neg)
        cv2.imwrite(filepath_neg, concat_neg)