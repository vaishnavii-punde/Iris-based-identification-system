import numpy as np
import tensorflow as tf
from siamese_model import build_siamese_model
from contrastive_loss import contrastive_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# ===============================
# Parameters
# ===============================
INPUT_SHAPE = (128, 128, 1)

# Use the best saved model
WEIGHTS_PATH = "models/best_siamese.keras"

# ===============================
# Load test data
# ===============================
pairs_test = np.load("pairs_test.npy", allow_pickle=True)
labels_test = np.load("labels_test.npy")

X1_test = np.array([p[0] for p in pairs_test], dtype=np.float32)
X2_test = np.array([p[1] for p in pairs_test], dtype=np.float32)

# ===============================
# Load model
# ===============================
# Option 1: Build architecture + load weights
try:
    model = build_siamese_model(INPUT_SHAPE)
    model.load_weights(WEIGHTS_PATH)
    print("Model weights loaded successfully!")
except:
    # Option 2: direct load (if TF saved full model)
    model = tf.keras.models.load_model(
        WEIGHTS_PATH,
        custom_objects={"contrastive_loss": contrastive_loss}
    )
    print("Full model loaded successfully!")

# ===============================
# Compute distances
# ===============================
distances = model.predict([X1_test, X2_test], batch_size=16).flatten()

# ===============================
# Dynamic threshold calculation
# ===============================
same_mask = labels_test == 1
diff_mask = labels_test == 0

same_mean = distances[same_mask].mean()
diff_mean = distances[diff_mask].mean()

THRESHOLD = (same_mean + diff_mean) / 2
np.save("dynamic_threshold.npy", THRESHOLD)
print(f"Dynamic Threshold: {THRESHOLD:.4f}")
print(f"Mean distance (same): {same_mean:.4f}")
print(f"Mean distance (different): {diff_mean:.4f}")

# ===============================
# Predict same/different
# ===============================
y_pred = (distances < THRESHOLD).astype(int)
y_true = labels_test

# ===============================
# Metrics
# ===============================
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred)

print("--------- Evaluation Results ---------")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print("Confusion Matrix:")
print(cm)
