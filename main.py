from model.pair_generator import load_all_images_from_group, make_self_supervised_pairs,make_similarity_pairs
from model.trainer import train_model
import sys
import os

from model.trainer import train_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.visualize import generate_similarity_matrix
from config import PREPROCESSED_DIR, DATA_DIR, TRIPLET_MODEL_PATH
from preprocessing.preprocess import batch_preprocess_folder
from pathlib import Path

from check_dataset import check_preprocessed_images, check_minimum_images_per_class, check_image_sizes
from model.triplet_generator import make_self_supervised_triplets, make_triplets_from_model
from model.triplet_trainer import train_triplet_model

def train_siamese_model(group, image_paths, my_path, coin_side):
    print("🔗 Erstelle Bildpaare...")
    #Pretraining (Self-Supervised)
    print(f"Gruppe {group} enthält {len(image_paths)} Bilder")
    ss_pairs, ss_labels = make_self_supervised_pairs(image_paths)
    print("🏋️ Starte Training...")
    model = train_model(ss_pairs, ss_labels, my_path, group, suffix = "pretrain", visualize=True, coin_side=coin_side)

    #3. Paarbildung
    print("🔗 Erstelle Similarity-Paare...")
    sim_pairs, sim_labels = make_similarity_pairs(model, image_paths, my_path, coin_side=coin_side, group=group )


    # 4. Finales Training auf similarity-pairs
    print("🏋️ Finales Training...")
    final_model = train_model(sim_pairs, sim_labels, my_path, group, suffix="final", visualize=True, coin_side=coin_side)

    #5. Visualisiere Beispielpaare & Ähnlichkeitsmatrix
    embedding_model = final_model.get_layer("embedding_model")
    generate_similarity_matrix(embedding_model, image_paths, my_path, group, model_type = 'siamese', coin_side=coin_side)
    # 6. Speichern
    result_dir = os.path.join(my_path, "result_models", "siamese_models")
    os.makedirs(result_dir, exist_ok=True)
    final_model.save(os.path.join(result_dir, f"siamese_model_{group}_{coin_side}_final.keras"))

    print(f"✅ Training für Gruppe {group} abgeschlossen.\n")
    return final_model

def triplet_model(group, image_paths, my_path, coin_side, mining = False):
    print(f"📦 Starte Triplet-Modellierung für Gruppe {group} ({coin_side})")

    # 1. Self-supervised Pretraining
    print("🧪 Phase 1: Self-supervised Training mit Augmentierungen...")
    anchors, positives, negatives = make_self_supervised_triplets(image_paths)
    model = train_triplet_model(anchors, positives, negatives, visualize=False, group=group, path=my_path, coin_side=coin_side)
    embedding_model = model.get_layer("Embedding")
    
    if mining:
        anchors, positives, negatives = make_self_supervised_triplets(image_paths, embedding_model=embedding_model)
        model = train_triplet_model(anchors, positives, negatives, visualize=False, group=group, path=my_path)
        embedding_model = model.get_layer("Embedding")
    
    # 2. Save Checkpoint
    model_path = os.path.join(TRIPLET_MODEL_PATH, f"triplet_model_{group}_{coin_side}_pretrained.keras")

    model.save(model_path)
    print(f"💾 Pretrained Modell gespeichert unter: {model_path}")

    # 3. Pseudo-supervised Fine-tuning
    print("🧪 Phase 2: Fine-Tuning mit Cosine-Triplets...")
    anchors, positives, negatives = make_triplets_from_model(model, image_paths, my_path, group, coin_side=coin_side, alpha=0.85)
    model = train_triplet_model(anchors, positives, negatives , visualize=True, group=group, path=my_path, coin_side=coin_side, margin=0.3)

    # 4. Save final model
    final_model_path = os.path.join(TRIPLET_MODEL_PATH, f"triplet_model_{group}_{coin_side}_triplet_final.keras")
    model.save(final_model_path)
    print(f"✅ Finales Triplet-Modell gespeichert unter: {final_model_path}")

    # 5. Optional: Ähnlichkeitsmatrix zur Visualisierung
    embedding_model = model.get_layer("Embedding")
    generate_similarity_matrix(embedding_model, image_paths, my_path, group, model_type = 'triplet', coin_side=coin_side)

    return model

def main(train_model = "siamese", coin_side = 'obverse', groups = ["A"], mining = False): #, "B", "C", "D", "E", "F", "G", "H"
    if coin_side in ['reverse', 'obverse']:
        data_dir = DATA_DIR + "/" + coin_side
        preprocessed_dir = PREPROCESSED_DIR + "/" + coin_side
    else:
        print(f"❌ Ungültige coin_side: {coin_side}. Bitte 'reverse' oder 'obverse' wählen.")
        return
    if not Path(preprocessed_dir).exists():
        print("🧹 Starte Preprocessing...")
        batch_preprocess_folder(data_dir, preprocessed_dir)
    else:
        print("✅ Preprocessed-Daten existieren bereits.")
    print("🔍 Überprüfe Preprocessed-Daten...")
    check_preprocessed_images(preprocessed_dir)
    check_minimum_images_per_class(preprocessed_dir, min_images=2)
    check_image_sizes(preprocessed_dir)
    print("📂 Lade Bilddaten...")
    
    my_path = os.path.dirname(os.path.abspath(__file__))
    print("Trainiere Modell für jede Gruppe separat...")
    for group in groups: 
        try:
            group_dir = os.path.join(preprocessed_dir, group)
            image_paths = load_all_images_from_group(group_dir)
            if len(image_paths) < 3:
                print(f"⚠️ Gruppe {group} hat zu wenige Bilder.")
                continue
            if train_model =="siamese":
                model = train_siamese_model(group, image_paths, my_path, coin_side)
            elif train_model == "triplet":
                model = triplet_model(group, image_paths, my_path, coin_side, mining=mining)
            else:
                print(f"❌ Unbekanntes Modell: {train_model}. Bitte 'siamese' oder 'triplet' wählen.")
                continue
            print(f"✅ Modell für Gruppe {group} trainiert.")

        except Exception as e:
            print(f"Fehler bei Gruppe {group}: {e}")

    print("Modelltraining abgeschlossen.")

if __name__ == "__main__":
    main()
#streamlit run stempel_ui_app.py