import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from siamese_model import build_siamese_model
from tensorflow import keras
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

import tensorflow as tf

def euclidean_distance(vectors):
    x, y = vectors
    return tf.math.abs(x - y)

# Enable unsafe deserialization (Lambda layers)

keras.config.enable_unsafe_deserialization()

# Load trained model

WEIGHTS_PATH = "models/best_siamese.keras"
model = build_siamese_model((128, 128, 1))

import gdown
import os

WEIGHTS_PATH = "model.weights.h5"  # whatever your filename is

if not os.path.exists(WEIGHTS_PATH):
    gdown.download(
        f"https://drive.google.com/uc?id=YOUR_FILE_ID_HERE",
        WEIGHTS_PATH,
        quiet=False
    )

model.load_weights(WEIGHTS_PATH)
model.load_weights(WEIGHTS_PATH)


# Preprocess function

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=-1)


# ORB Feature Extraction

def compute_orb(img):
    orb = cv2.ORB_create(500)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

def draw_orb_matches(img1, img2, kp1, kp2, des1, des2):
    if des1 is None or des2 is None:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)


# LBP Feature Extraction

def compute_lbp_histogram(img, P=8, R=1):
    lbp = local_binary_pattern(img, P, R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, P + 3),
                             range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def lbp_distance(img1, img2):
    hist1 = compute_lbp_histogram(img1)
    hist2 = compute_lbp_histogram(img2)
    return cv2.compareHist(hist1.astype("float32"), hist2.astype("float32"), cv2.HISTCMP_CHISQR)

# Siamese distance

def get_distance(imgA, imgB):
    A = np.expand_dims(preprocess(imgA), axis=0)
    B = np.expand_dims(preprocess(imgB), axis=0)
    distance = model.predict([A, B])[0][0]
    return float(distance)


# Load dynamic threshold

THRESHOLD = float(np.load("dynamic_threshold.npy"))
st.info(f"Using Dynamic Threshold: **{THRESHOLD:.4f}**")


# UI

st.title("👁️ Iris-Based Person Identification System")
st.write("Upload images for Person 1 and Person 2")

col1, col2 = st.columns(2)

with col1:
    p1_files = st.file_uploader("Upload Person 1 Images", type=["png","jpg","jpeg"], accept_multiple_files=True)

with col2:
    p2_files = st.file_uploader("Upload Person 2 Images", type=["png","jpg","jpeg"], accept_multiple_files=True)

if p1_files and p2_files:

    st.subheader("🔍 Computing Distances...")
    all_final_scores = []
    all_distances = []
    all_matches = []

    for f1 in p1_files:
        img1 = cv2.imdecode(np.frombuffer(f1.read(), np.uint8), cv2.IMREAD_COLOR)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        kp1, des1 = compute_orb(gray1)

        for f2 in p2_files:
            img2 = cv2.imdecode(np.frombuffer(f2.read(), np.uint8), cv2.IMREAD_COLOR)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            kp2, des2 = compute_orb(gray2)

            # Siamese distance
            siam_dist = get_distance(img1, img2)

            # ORB match ratio (normalized)
            if des1 is not None and des2 is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                orb_score = len(matches) / max(len(kp1), len(kp2), 1)
                orb_dist = 1 - orb_score  # convert to distance
            else:
                orb_dist = 1.0

            # LBP distance
            lbp_dist = lbp_distance(gray1, gray2)

            # Final weighted score
            final_score = 0.5 * siam_dist + 0.3 * lbp_dist + 0.2 * orb_dist
            all_final_scores.append(final_score)
            all_distances.append(siam_dist)

            # Draw ORB matches
            match_img = draw_orb_matches(gray1, gray2, kp1, kp2, des1, des2)
            all_matches.append((f1.name, f2.name, match_img, siam_dist, final_score))

    avg_final_score = np.mean(all_final_scores)
    st.metric("Average Final Score", f"{avg_final_score:.4f}")

    
    # Final prediction

    if avg_final_score > THRESHOLD:
        st.success("❌ DIFFERENT PERSON")
    else:
        st.error("✅ SAME PERSON")


  
    # ORB Matches
   
    st.subheader("🔍 ORB Feature Matches")
    for name1, name2, match_img, d, final_score in all_matches:
        st.write(f"**{name1} ↔ {name2}** — Siamese: `{d:.4f}`, Final Score: `{final_score:.4f}`")
        if match_img is not None:
            st.image(match_img, channels="BGR")
        else:
            st.write("⚠️ No ORB features detected.")

else:
    st.info("Upload images for both persons to proceed.")


import streamlit as st
import gdown
import os

WEIGHTS_PATH = "best_siamese.keras"

if not os.path.exists(WEIGHTS_PATH):
    file_id = "16IIXxoJnQxjobCw1M20vKUrZkg51G5fN"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(url, WEIGHTS_PATH, quiet=False)

model = build_siamese_model((128, 128, 1))
model.load_weights(WEIGHTS_PATH)
