from model.siamese_model import build_siamese_model
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras import metrics
from config import BATCH_SIZE, EPOCHS
import sys
import os
from utils.visualize import save_prediction_pairs, save_similarity_matrix, visualize_predictions
from tensorflow.keras.utils import Sequence
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class PairGenerator(Sequence):
    def __init__(self, X1, X2, y, batch_size=32, shuffle=True):
        self.X1 = X1
        self.X2 = X2
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(y))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_X1 = self.X1[batch_idx].astype("float32")
        batch_X2 = self.X2[batch_idx].astype("float32")
        batch_y = self.y[batch_idx]
        return (batch_X1, batch_X2), batch_y


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def train_model(pairs, labels, path, group=None, suffix="", visualize=True, coin_side="reverse"):
    X1 = pairs[:, 0]
    X2 = pairs[:, 1]
    y = labels
    # 1. Aufteilen in Training und Test
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
        X1, X2, y, test_size=0.3, random_state=42, stratify=y
    )
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
        X1_train, X2_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    model = build_siamese_model()
    print(f"Summary des Modells für Gruppe {group} ({suffix}):")
    model.summary()
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy", metrics.AUC(name="auc")]
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    train_gen = PairGenerator(X1_train, X2_train, y_train, batch_size=BATCH_SIZE)
    val_gen = PairGenerator(X1_val, X2_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    test_gen = PairGenerator(X1_test, X2_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[early_stopping]
    )

    print("Evaluation auf Testset:")
    loss, accuracy, auc = model.evaluate([X1_test, X2_test], y_test)
    print(f"📈 Test Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

    # Optional, falls similarity_matrix verfügbar ist (z. B. aus make_similarity_pairs)
    if visualize:
        print("Start Visualizing")
        visualize_predictions(model, X1_val, X2_val, y_val, path, group, model_type = 'siamese', coin_side=coin_side)
        print("Speichere Vorhersagepaare...")
        save_prediction_pairs(model, X1_val, X2_val, y_val, path, group, model_type = 'siamese'  , coin_side=coin_side)

        if hasattr(train_model, "similarity_matrix"):
            print("Speichere Ähnlichkeitsmatrix...")
            save_similarity_matrix(train_model.similarity_matrix, path, group, model_type = 'siamese', coin_side=coin_side)


    return model