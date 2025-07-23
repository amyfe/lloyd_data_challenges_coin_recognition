# model/triplet_model.py
from tensorflow.keras import layers, Model, Input
from config import INPUT_SHAPE
import tensorflow as tf

def build_embedding_network(input_shape=INPUT_SHAPE):
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(64, (3,3), activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128)(x)  # Embedding size
    return Model(inputs, x, name="Embedding")