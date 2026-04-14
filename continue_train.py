# continue_train.py — Continue training Siamese model

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from siamese_model import build_siamese_model
import os

# ===============================
# Hyperparameters
# ===============================
BATCH_SIZE = 16
EPOCHS = 10   # train additional 5-10 epochs
LEARNING_RATE = 1e-4  # lower LR for fine-tuning

# ===============================
# Paths
# ===============================
BEST_WEIGHTS_PATH = "models/best_siamese.keras"

# ===============================
# Load Data
# ===============================
pairs_train = np.load("pairs_train.npy", allow_pickle=True)
pairs_test = np.load("pairs_test.npy", allow_pickle=True)
labels_train = np.load("labels_train.npy")
labels_test = np.load("labels_test.npy")

print("Train samples:", len(pairs_train))
print("Test samples :", len(pairs_test))

# ===============================
# Split pairs
# ===============================
def split_pairs(pairs):
    X1 = np.array([p[0] for p in pairs], dtype=np.float32)
    X2 = np.array([p[1] for p in pairs], dtype=np.float32)
    return X1, X2

X1_train, X2_train = split_pairs(pairs_train)
X1_test, X2_test = split_pairs(pairs_test)

# ===============================
# TF Datasets
# ===============================
train_dataset = tf.data.Dataset.from_tensor_slices(((X1_train, X2_train), labels_train))
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices(((X1_test, X2_test), labels_test))
val_dataset = val_dataset.batch(BATCH_SIZE)

# ===============================
# Build model
# ===============================
model = build_siamese_model((128,128,1), learning_rate=LEARNING_RATE)
print("Model built successfully!")

# Load previous weights
model.load_weights(BEST_WEIGHTS_PATH)
print("Previous weights loaded successfully!")

# ===============================
# Callbacks
# ===============================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

os.makedirs("models", exist_ok=True)
checkpoint = ModelCheckpoint(
    "models/best_siamese_continue.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

callbacks = [early_stop, reduce_lr, checkpoint]

# ===============================
# Continue Training
# ===============================
print("Starting continued training...")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ===============================
# Save final weights
# ===============================
model.save_weights("models/siamese_model_final.weights.h5")
model.save("models/siamese_model_final.keras")
print("Continued training finished! Weights saved successfully!")
