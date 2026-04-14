import cv2
import os

# Input and output folders
input_folder = "dataset"
output_folder = "dataset_processed"
target_size = (128, 128)

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Allowed image extensions (added BMP)
valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]

for person in os.listdir(input_folder):
    person_path = os.path.join(input_folder, person)
    if not os.path.isdir(person_path):
        continue

    output_person_path = os.path.join(output_folder, person)
    os.makedirs(output_person_path, exist_ok=True)

    for eye_side in ["left", "right"]:
        eye_path = os.path.join(person_path, eye_side)
        if not os.path.exists(eye_path):
            print(f"Warning: Folder not found {eye_path}")
            continue

        output_eye_path = os.path.join(output_person_path, eye_side)
        os.makedirs(output_eye_path, exist_ok=True)

        for img_name in os.listdir(eye_path):

            # Skip non-image files (like Thumbs.db)
            if img_name.lower().endswith(".db"):
                continue

            if not any(img_name.endswith(ext) for ext in valid_exts):
                print(f"Skipping non-image file: {img_name}")
                continue

            img_path = os.path.join(eye_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue

            img = cv2.resize(img, target_size)
            cv2.imwrite(os.path.join(output_eye_path, img_name), img)

print("Preprocessing complete. Check dataset_processed folder.")
