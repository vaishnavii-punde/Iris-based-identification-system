# contrastive_loss.py — OLD WORKING VERSION

import tensorflow as tf

def contrastive_loss(y_true, y_pred, margin=1.0):
    """
    Contrastive loss for Siamese network with scalar similarity output.
    y_pred is sigmoid output (0 to 1) — NOT a vector.
    """
    y_true = tf.cast(y_true, tf.float32)

    # y_pred = predicted similarity (0 to 1)
    # Convert to distance for contrastive loss
    distance = y_pred

    positive_loss = y_true * tf.square(distance)                 # same person
    negative_loss = (1 - y_true) * tf.square(tf.maximum(margin - distance, 0))  # different person

    return tf.reduce_mean(positive_loss + negative_loss)
