import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from siamese_model import build_siamese_model

# -----------------------------
# 1️⃣ Load the model
# -----------------------------
model = build_siamese_model(input_shape=(128, 128, 1))
model.load_weights("models/siamese_model.weights.h5")
print("✅ Model loaded successfully!")

# -----------------------------
# 2️⃣ Load test data
# -----------------------------
pairs_test = np.load("pairs_test.npy")
labels_test = np.load("labels_test.npy")
print(f"Test pairs: {len(labels_test)}")

X1_test = pairs_test[:, 0]
X2_test = pairs_test[:, 1]

# -----------------------------
# 3️⃣ Predict similarity scores
# -----------------------------
similarity_scores = model.predict([X1_test, X2_test], batch_size=32)
similarity_scores = similarity_scores.flatten()

same_scores = similarity_scores[labels_test == 1]
diff_scores = similarity_scores[labels_test == 0]

# -----------------------------
# 4️⃣ Plot distributions
# -----------------------------
plt.figure(figsize=(10,6))
plt.hist(same_scores, bins=50, alpha=0.7, label="Same person")
plt.hist(diff_scores, bins=50, alpha=0.7, label="Different person")
plt.xlabel("Similarity Score")
plt.ylabel("Number of Pairs")
plt.title("Similarity Score Distribution")
plt.legend()
plt.show()

# -----------------------------
# 5️⃣ Suggest a threshold
# -----------------------------
best_f1 = 0
best_threshold = 0

for threshold in np.linspace(0, 1, 101):
    preds = (similarity_scores >= threshold).astype(int)
    tp = np.sum((preds == 1) & (labels_test == 1))
    fp = np.sum((preds == 1) & (labels_test == 0))
    fn = np.sum((preds == 0) & (labels_test == 1))
    
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"✅ Suggested threshold: {best_threshold:.2f} with F1-score: {best_f1:.4f}")
