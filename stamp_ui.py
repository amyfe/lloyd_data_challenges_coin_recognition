import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model
from config import IMAGE_SIZE, TRIPLET_MODEL_PATH, SIAMESE_MODEL_PATH
import os
import streamlit.components.v1 as components
from keras.config import enable_unsafe_deserialization
import top_similar

# Run with:  streamlit run stamp_ui.py

@st.cache_data
def load_images(image_paths):
    images = []
    names = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is not None:
            img = cv2.resize(img, IMAGE_SIZE)
            img = img.astype("float32") / 255.0
            images.append(img)
            names.append(path.name)
    return np.array(images), names

@st.cache_resource
def get_embeddings(model_path, image_paths):
    from keras.config import enable_unsafe_deserialization
    enable_unsafe_deserialization()
    model = load_model(model_path)
    print(f"🔍 Lade Modell:", model.summary())
    embedding_model = model.get_layer("embedding_model")

    images, names = load_images(image_paths)
    embeddings = embedding_model.predict(images, verbose=0)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings, names, images

@st.cache_resource
def load_my_model():
    enable_unsafe_deserialization()
    model_path = os.path.join(SIAMESE_MODEL_PATH, "siamese_model_A_reverse_final.keras")

    model = load_model(model_path, compile=False)
    print("Heeere the summary",model.summary())
    if not os.path.exists(model_path):
        st.error(f"Modell nicht gefunden: {model_path}")
        return None
    return load_model(model_path)


def get_image_path( visualisation_type,model,group, coin_side):
    visualisation_path = Path(f"visuals/{model}_models/{visualisation_type}")

    if visualisation_type  == "confusion_matrix":
        visualisation_path = os.path.join(visualisation_path, f"confusion_matrix_{coin_side}_group_{group}.png")
        if not Path(visualisation_path).exists():
            st.error(f"Visualisierung nicht gefunden: {visualisation_path}")
        else:
            st.image(visualisation_path, caption="Confusion Matrix")
    elif visualisation_type == "similarity_distribution":
        visualisation_path = os.path.join(visualisation_path, f"cosine_distribution-{coin_side}_group_{group}.png")
        if not Path(visualisation_path).exists():
            st.error(f"Visualisierung nicht gefunden: {visualisation_path}")
        else:
            st.image(visualisation_path, caption="Similarity Distribution")
    elif visualisation_type == "similarity_matrix":
        visualisation_path = os.path.join(visualisation_path, f"sim_matrix_{coin_side}_{group}.png")
        if not Path(visualisation_path).exists():
            st.error(f"Visualisierung nicht gefunden: {visualisation_path}")
        else:
            st.image(visualisation_path, caption="Similarity Matrix")
    elif visualisation_type == "t_sne":
        visualisation_path = os.path.join(visualisation_path, f"confusion_matrix_group_{coin_side}_{group}.png")
        if not Path(visualisation_path).exists():
            st.error(f"Visualisierung nicht gefunden: {visualisation_path}")
        else:
            st.image(visualisation_path, caption="t-SNE Visualization")
st.title("🔎 Stempelähnlichkeit prüfen")

# Auswahl der Gruppe & Seite
group = st.selectbox("Wähle eine Gruppe", list("ABCDEFGH"))
coin_side = st.selectbox("Wähle Münzseite", ["reverse", "obverse"])
model = st.selectbox("Wähle Model", ["siamese", "triplet"])
task = st.selectbox("Wähle Anzeige", ["Visualisierung", "Vergleich"])

# Lade Bilder
image_folder = Path(f"data/{coin_side}/{group}")
image_paths = list(image_folder.glob("*.jpg"))
image_files = [p.name for p in image_paths]

if task == "Vergleich":
    vergleich_bild = st.selectbox("Wähle Vergleichsbild", image_files, index=0)
    top_n = st.slider("Top N ähnliche Münzen", 1, len(image_files)//2, 5)
    st.header(f"Vergleichsbild: {vergleich_bild}")
    vergleich_path = os.path.join(image_folder, vergleich_bild)
    st.image(vergleich_path, caption=f"🔍 Vergleichsbild: {vergleich_bild}", width=300)

    st.header("🔍 Ähnliche Münzen")
    sim_images = top_similar.top_similar(name=vergleich_bild, model=model, group=group, coin_side=coin_side, top_n=top_n)
    if sim_images:
        cols = st.columns(5)
        for i, (name, similarity) in enumerate(sim_images):
            with cols[i % 5]:
                img_path = os.path.join(image_folder, name)
                st.image(img_path, caption=f"{name}\nÄhnlichkeit: {similarity:.4f}", width=150)
    else:
        st.warning("Keine ähnlichen Münzen gefunden.")



if task == "Visualisierung":
    
    visualisation_type = st.selectbox("Wähle Visualisierungstyp", ["confusion_matrix","similarity_distribution","similarity_matrix","t_sne","Alles oben","Network Graph"])

    st.header(f"Visualisierung:")
    visualisation_path = Path(f"visuals/{model}_models/{visualisation_type}")

    if visualisation_type == "Alles oben":
        visualisation_types = ["confusion_matrix", "similarity_distribution", "similarity_matrix", "t_sne"]
        for v_type in visualisation_types:
            get_image_path(v_type,model, group, coin_side)
    elif visualisation_type == "Network Graph":
        network_graph_path = os.path.join("graphs", f'network_graph_{model}_{coin_side}_{group}.html')


        # Read and render it
        with open(network_graph_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Display the HTML
        components.html(html_content, height=1200, scrolling=True)
    else:
        get_image_path(visualisation_type,model, group, coin_side)



_ = """
model_path = os.path.join(SIAMESE_MODEL_PATH, f"siamese_model_{group}_{coin_side}_final.keras")
model = load_my_model()
embeddings, names, images = get_embeddings(model_path, image_paths)

# Auswahl eines Bildes
idx = st.selectbox("Wähle eine Referenzmünze", list(range(len(names))), format_func=lambda i: names[i])
ref_embedding = embeddings[idx:idx+1]

# Ähnlichkeit berechnen
sims = cosine_similarity(ref_embedding, embeddings).flatten()
sorted_indices = np.argsort(-sims)  # absteigend sortiert

# Threshold-Regler
threshold = st.slider("Ähnlichkeitsschwelle", 0.90, 1.0, 0.995, 0.0005)

# Anzeige
st.image(images[idx], caption=f"🔍 Referenz: {names[idx]}", width=300)
st.markdown("### Ähnliche Münzen:")

cols = st.columns(5)
shown = 0
for i in sorted_indices:
    if i == idx:
        continue
    if sims[i] >= threshold:
        with cols[shown % 5]:
            st.image(images[i], caption=f"{names[i]}\nSim: {sims[i]:.4f}", width=150)
        shown += 1
    if shown >= 20:
        break

if shown == 0:
    st.warning("Keine ähnlichen Münzen über dem Schwellenwert gefunden.")
"""

_ = """
# Cluster-Analyse
st.markdown("---")
if st.button("🔍 Cluster analysieren (DBSCAN)"):
    clustering = DBSCAN(eps=0.02, min_samples=2, metric='cosine').fit(embeddings)
    labels = clustering.labels_

    st.subheader("🧠 Gefundene Stempel-Cluster")
    st.write(f"Anzahl Cluster (ohne Rauschen): {len(set(labels)) - (1 if -1 in labels else 0)}")

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', s=40)
    ax.set_title("Embedding-Clustering via PCA + DBSCAN")
    st.pyplot(fig)

# Ähnlichkeitsverteilung
st.markdown("---")
if st.button("📊 Ähnlichkeitsverteilung anzeigen"):
    sims_all = cosine_similarity(embeddings)
    n = len(sims_all)
    all_sims = sims_all[np.triu_indices(n, k=1)]  # nur obere Hälfte (ohne Diagonale)

    fig, ax = plt.subplots()
    ax.hist(all_sims, bins=100, color='skyblue', edgecolor='k')
    ax.set_title("Histogramm der Cosine-Similarities aller Münzpaare")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Anzahl Paare")
    st.pyplot(fig)

    st.info("Je weiter rechts die Verteilung liegt, desto ähnlicher sind sich die Embeddings. Eine bimodale Verteilung könnte auf echte Gruppen/Stempel hinweisen.")
"""