# siamese_model.py — OLD WORKING VERSION

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from contrastive_loss import contrastive_loss

def build_base_network(input_shape):
    inp = Input(shape=input_shape)

    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)

    return models.Model(inp, x)

def build_siamese_model(input_shape, learning_rate=0.001):
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    base_net = build_base_network(input_shape)

    feat_a = base_net(input_a)
    feat_b = base_net(input_b)

    # L1 distance
    distance = layers.Lambda(lambda t: tf.abs(t[0] - t[1]))([feat_a, feat_b])

    # Dense → scalar sigmoid (OLD WORKING)
    output = layers.Dense(1, activation="sigmoid")(distance)

    model = models.Model([input_a, input_b], output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=contrastive_loss,
        metrics=["accuracy"]
    )

    return model
