# create_pairs.py — OLD WORKING VERSION (balanced pairs)

import os
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split

input_folder = "dataset_processed"
img_size = (128, 128)

def load_images():
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

def create_pairs(data):
    persons = list(data.keys())
    same_pairs = []
    diff_pairs = []

    # Create same-person pairs
    for person in persons:
        imgs = data[person]
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                same_pairs.append((imgs[i], imgs[j], 1))

    # Create different-person pairs
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            p1 = persons[i]
            p2 = persons[j]

            img1 = random.choice(data[p1])
            img2 = random.choice(data[p2])

            diff_pairs.append((img1, img2, 0))

    # Balance
    min_count = min(len(same_pairs), len(diff_pairs))
    same_pairs = random.sample(same_pairs, min_count)
    diff_pairs = random.sample(diff_pairs, min_count)

    print(f"Same pairs: {len(same_pairs)}")
    print(f"Diff pairs: {len(diff_pairs)}")

    all_pairs = same_pairs + diff_pairs
    random.shuffle(all_pairs)

    # Split
    train_pairs, test_pairs = train_test_split(all_pairs, test_size=0.2)

    X_train = [(p[0], p[1]) for p in train_pairs]
    y_train = np.array([p[2] for p in train_pairs])

    X_test = [(p[0], p[1]) for p in test_pairs]
    y_test = np.array([p[2] for p in test_pairs])

    return X_train, X_test, y_train, y_test

data = load_images()
X_train, X_test, y_train, y_test = create_pairs(data)

np.save("pairs_train.npy", np.array(X_train, dtype=object))
np.save("pairs_test.npy", np.array(X_test, dtype=object))
np.save("labels_train.npy", y_train)
np.save("labels_test.npy", y_test)

print("Saved dataset successfully!")
