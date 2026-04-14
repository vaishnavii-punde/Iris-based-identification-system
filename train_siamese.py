# train_siamese.py — OLD WORKING TRAINER

import numpy as np
import tensorflow as tf
from siamese_model import build_siamese_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

# -------------------------
# Load Data
# -------------------------
pairs_train = np.load("pairs_train.npy", allow_pickle=True)
pairs_test = np.load("pairs_test.npy", allow_pickle=True)
labels_train = np.load("labels_train.npy")
labels_test = np.load("labels_test.npy")

def split_pairs(pairs):
    X1 = np.array([p[0] for p in pairs])
    X2 = np.array([p[1] for p in pairs])
    return X1, X2

X1_train, X2_train = split_pairs(pairs_train)
X1_test, X2_test = split_pairs(pairs_test)

train_dataset = tf.data.Dataset.from_tensor_slices(((X1_train, X2_train), labels_train))
train_dataset = train_dataset.shuffle(5000).batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices(((X1_test, X2_test), labels_test))
val_dataset = val_dataset.batch(32)

# -------------------------
# Build model
# -------------------------
model = build_siamese_model((128,128,1))
model.summary()

# -------------------------
# Callbacks
# -------------------------
os.makedirs("models", exist_ok=True)

checkpoint = ModelCheckpoint(
    "models/best_siamese.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-5,
    verbose=1
)

# -------------------------
# Train
# -------------------------
print("Starting training...")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    callbacks=[checkpoint, early, reduce_lr]
)

model.save("models/final_siamese.keras")
