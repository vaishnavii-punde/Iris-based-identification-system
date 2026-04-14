import numpy as np

def balance_pairs(pairs, labels):
    pairs = np.array(pairs, dtype=object)
    labels = np.array(labels)

    # separate positive & negative
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    print("Before balancing:")
    print("Positive pairs:", len(pos_idx))
    print("Negative pairs:", len(neg_idx))

    # choose equal count
    min_count = min(len(pos_idx), len(neg_idx))

    pos_idx = np.random.choice(pos_idx, min_count, replace=False)
    neg_idx = np.random.choice(neg_idx, min_count, replace=False)

    final_idx = np.concatenate([pos_idx, neg_idx])
    np.random.shuffle(final_idx)

    return pairs[final_idx], labels[final_idx]


# -----------------------------
# Load original pair files
# -----------------------------
pairs_train = np.load("pairs_train.npy", allow_pickle=True)
labels_train = np.load("labels_train.npy")

pairs_test = np.load("pairs_test.npy", allow_pickle=True)
labels_test = np.load("labels_test.npy")

# -----------------------------
# Balance the pairs
# -----------------------------
pairs_train_bal, labels_train_bal = balance_pairs(pairs_train, labels_train)
pairs_test_bal, labels_test_bal = balance_pairs(pairs_test, labels_test)

# -----------------------------
# Save new balanced versions
# -----------------------------
np.save("pairs_train_bal.npy", pairs_train_bal)
np.save("labels_train_bal.npy", labels_train_bal)

np.save("pairs_test_bal.npy", pairs_test_bal)
np.save("labels_test_bal.npy", labels_test_bal)

print("\nBalanced datasets saved successfully!")
