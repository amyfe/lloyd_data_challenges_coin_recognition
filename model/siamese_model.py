import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from config import INPUT_SHAPE

def build_embedding_model():
    inputs = Input(shape=INPUT_SHAPE)
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    return Model(inputs, x, name="embedding_model")

def build_siamese_model():
    input_a = Input(shape=INPUT_SHAPE)
    input_b = Input(shape=INPUT_SHAPE)

    embedding_model = build_embedding_model()
    emb_a = embedding_model(input_a)
    emb_b = embedding_model(input_b)

    l1 = layers.Lambda(
        lambda tensors: tf.abs(tensors[0] - tensors[1]),
        output_shape=lambda input_shapes: input_shapes[0]
    )([emb_a, emb_b])
    output = layers.Dense(1, activation="sigmoid")(l1)

    model = Model([input_a, input_b], output, name="siamese_network")
    model.summary()
    return model