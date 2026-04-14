# generate_test_pairs.py — OLD WORKING VERSION

import os
import numpy as np
import cv2
import random

input_folder = "dataset_processed"
img_size = (128, 128)

def load_data():
    data = {}
    for person in sorted(os.listdir(input_folder)):
        person_path = os.path.join(input_folder, person)
        if not os.path.isdir(person_path):
            continue

        images = []
        for file in os.listdir(person_path):
            if file.lower().endswith(("png", "jpg", "jpeg")):
                img = cv2.imread(os.path.join(person_path, file), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, img_size)
                img = img.astype("float32") / 255.0
                images.append(img)

        if len(images) > 0:
            data[person] = images
    return data

data = load_data()
persons = list(data.keys())

pairs = []
labels = []

for person in persons:
    imgs = data[person]
    if len(imgs) < 2:
        continue

    # same pair
    p1, p2 = random.sample(imgs, 2)
    pairs.append((p1, p2))
    labels.append(1)

    # one different pair
    other = random.choice([p for p in persons if p != person])
    img_other = random.choice(data[other])
    pairs.append((p1, img_other))
    labels.append(0)

np.save("test_pairs.npy", np.array(pairs, dtype=object))
np.save("test_labels.npy", np.array(labels))

print("Test pairs saved!")
