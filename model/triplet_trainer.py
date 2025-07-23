import tensorflow as tf
from model.triplet_model import  build_embedding_network # identisch mit dem, was auch im Siamese verwendet wird
from config import BATCH_SIZE, EPOCHS
from sklearn.model_selection import train_test_split
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf

from utils.visualize import save_prediction_pairs, save_similarity_matrix, save_triplet_prediction_pairs, visualize_predictions, visualize_triplet_embeddings


def make_triplet_dataset(anchors, positives, negatives, batch_size=32, shuffle=True):
    labels = np.zeros(len(anchors), dtype=np.float32)  # dummy labels
    dataset = tf.data.Dataset.from_tensor_slices(((anchors, positives, negatives), labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(anchors))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def triplet_accuracy(y_true, y_pred):
    embedding_dim = y_pred.shape[1] // 3
    anchor   = y_pred[:, 0:embedding_dim]
    positive = y_pred[:, embedding_dim:2*embedding_dim]
    negative = y_pred[:, 2*embedding_dim:3*embedding_dim]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    return tf.reduce_mean(tf.cast(pos_dist < neg_dist, tf.float32))
 # model/losses.py

def triplet_loss(margin=0.3):
    def loss(y_true, y_pred):
        embedding_dim = y_pred.shape[1] // 3
        anchor   = y_pred[:, 0:embedding_dim]
        positive = y_pred[:, embedding_dim:2*embedding_dim]
        negative = y_pred[:, 2*embedding_dim:3*embedding_dim]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        basic_loss = pos_dist - neg_dist + margin
        return tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss

def build_triplet_model(input_shape):
    embedding_model = build_embedding_network(input_shape)
    input_anchor = tf.keras.Input(shape=input_shape, name="anchor")
    input_positive = tf.keras.Input(shape=input_shape, name="positive")
    input_negative = tf.keras.Input(shape=input_shape, name="negative")

    emb_anchor = embedding_model(input_anchor)
    emb_positive = embedding_model(input_positive)
    emb_negative = embedding_model(input_negative)

    merged = tf.keras.layers.Concatenate(axis=1)([emb_anchor, emb_positive, emb_negative])
    model = tf.keras.Model(inputs=[input_anchor, input_positive, input_negative], outputs=merged)
    return model, embedding_model

def train_triplet_model(anchors, positives, negatives, visualize=True, group=None, path=None, coin_side="reverse", margin=0.2):
    anchors = np.array(anchors).astype("float32")
    positives = np.array(positives).astype("float32")
    negatives = np.array(negatives).astype("float32")

    X_train, X_val, pos_train, pos_val, neg_train, neg_val = train_test_split(
        anchors, positives, negatives, test_size=0.2, random_state=42
    )
    train_ds = make_triplet_dataset(X_train, pos_train, neg_train, batch_size=BATCH_SIZE)
    val_ds = make_triplet_dataset(X_val, pos_val, neg_val, batch_size=BATCH_SIZE, shuffle=False)

    input_shape = anchors[0].shape
    model, embedding_model = build_triplet_model(input_shape)
    model.summary()
    model.compile(optimizer='adam', loss=triplet_loss(margin=margin), metrics=[triplet_accuracy])

    # Add callbacks here (see below)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    if visualize:
        print("Start Visualizing")
        visualize_triplet_embeddings(embedding_model, X_val, pos_val, neg_val, path=path, group=group, coin_side=coin_side)
        print("Speichere Vorhersagepaare...")
        save_triplet_prediction_pairs(embedding_model, X_val, pos_val, neg_val, path=path, group=group, coin_side=coin_side)

        if hasattr(model, "similarity_matrix"):
            print("Speichere Ähnlichkeitsmatrix...")
            save_similarity_matrix(model.similarity_matrix, path, group, model_type = 'triplet', coin_side=coin_side)
    return model  # nur das embedding_model wird verwendet und gespeichert
