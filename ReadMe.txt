# 🧠 Stempelgleichheitserkennung mit Triplet Learning

Dieses Projekt beschäftigt sich mit der automatisierten Erkennung von **ähnlichen oder identischen Stempeln** auf Münzbildern. Durch den Einsatz von **Siamese- und Triplet-Netzwerken** mit **Cosine Similarity**, **Triplet Loss** und **t-SNE Visualisierung** wird ein Modell trainiert, das feine Unterschiede im Stempelbild erkennen kann.

---

## 📦 Inhalte

- 📁 `main.py` – Haupttraining inkl. selbst- und pseudogelabelter Siamesischer- & Triplet-Erstellung
- 📁 `model/` – Architektur des Embedding-Modells
- 📁 `utils/` – Visualisierungen, Evaluation, Triplet-Erstellung
- 📁 `config.py` – zentrale Parameter (Batch Size, Image Size, Pfade, etc.)
- 📁 `visuals/` – explorative Analyse und Tests
- 📁 `data/` – Pfade zu Bilddaten (nicht enthalten) -> WICHTIG: beim Testen muss die Datei noch hinzugefügt werden
- 📁 `jupyter_notebook/` – Embeddings für CoinCLIP 
- 📁 `embeddings_result/` – Resultate der Modelle für jede Gruppe und jede Münzseite für beide Modelle
- 📁 `preprocessing/` – Daten werden durch Ausführung von main.py vorverarbeitet (Hintergrundentfernung, Kontrastfarben,...)
- 📁 `stamp_ui`, `top_similar`, `network_graph` – Frontend inkl Datenvisualisierungen

---

## 🚀 Ziel

Ziel ist es, **Münzen mit gleichem Stempel** zuverlässig zu erkennen – auch bei feinen Abweichungen, perspektivischen Verzerrungen oder Patina.

---

## 🧪 Vorgehensweise

### 1. Preprocessing
- Bilder werden auf 224×224 oder 500×500 skaliert (je nach Experiment)
- Farbnormalisierung und Augmentierungen (Rotation, Flip, Zoom)

### 2. Modell: Embedding-Netzwerk
- CNN-basiertes Modell zur Extraktion von **Embedding-Vektoren**
- Siamese- und Triplet-Struktur verwendet gleiche Gewichte (Weight Sharing)

### 3. Training in 2 Phasen
#### 🔹 Phase 1: Selbstüberwachtes Lernen
- Contractive Loss / Triplets werden durch **Augmentationen** erzeugt
- Training mit klassischem **Contractive Loss /Triplet Loss**

#### 🔹 Phase 2: Pseudo-Supervised Fine-Tuning (Bei Triplet)
- Ähnliche Paare werden mithilfe von **Cosine Similarity** innerhalb einer Gruppe identifiziert
- Feintuning mit stärkeren negativen Beispielen (Triplet Mining, wenn mining = True)

#### 🔹 Phase 3: Fine-Tuning
-Embedings der vorigen Phase mit Coin-Clip Embeddings konkatenieren
- Ähnliche Paare werden mithilfe von **Cosine Similarity** innerhalb einer Gruppe identifiziert

### 4. Evaluation
- Accuracy, Loss, AUC auf Trainings- und Validierungsdaten
- Visualisierung der **t-SNE Embeddings** nach dem Training
- Ähnlichkeitsmatrizen zur qualitativen Analyse

---

## 📊 Beispielhafte Ergebnisse

| Metrik         | Wert       |
|----------------|------------|
| Accuracy       | 0.9388     |
| AUC            | 0.9839     |
| Loss           | 0.2024     |
| Val Accuracy   | 0.9200     |
| Val AUC        | 0.9827     |
| Val Loss       | 0.2296     |

---

## 🧮 Verwendete Pakete

Siehe requierements.txt

---

## 📂 Datenstruktur

```plaintext
project/
│
├── data/
    ├── reverse (analog auch obverse)
    │   ├── A/
    │   │   ├── img001.jpg
    │   │   └── ...
│   
├── model/
│   └── triplet_model-files
    └── siamese_model-files
├── utils/
│   └── visualize.py
├── main.py
└── config.py


### Start der Modelle:
1. requierements müssen vorhanden sein (siehe requierements.txt)
2. data muss in dem richtigen Ordner sein
3. main.py laufen lassen, Bedeutung der Parameter:
Auswahl des Models [str]: train_model = "siamese" oder "triplet",
Seite der Münze [str]: coin_side = 'obverse' oder "reverse",
Gruppen [list]: z.B. ["A", "B"]
Bei Triplet, ob man noch mit Mining das Model berechnen möchte [bool] : mining = False