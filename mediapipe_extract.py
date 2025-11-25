import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

mp_hands = mp.solutions.hands

def extract_landmarks_from_image(img_path):
    """Return landmark array of shape (21, 3) or None if hand not detected."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1
    ) as hands:
        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            return None

        lm = results.multi_hand_landmarks[0]
        coords = []
        for point in lm.landmark:
            coords.append([point.x, point.y, point.z])

        return np.array(coords)


def process_dataset(raw_dir, save_prefix, limit=100):
    X, y = [], []
    classes = sorted(os.listdir(raw_dir))

    for label in classes:
        folder = os.path.join(raw_dir, label)
        if not os.path.isdir(folder):
            continue

        print(f"\nProcessing class: {label}")

        images = sorted(os.listdir(folder))[:limit]

        for img_name in tqdm(images, desc=f"{label}"):
            img_path = os.path.join(folder, img_name)
            coords = extract_landmarks_from_image(img_path)
            if coords is not None:
                X.append(coords)
                y.append(label)

    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)

    np.save(save_prefix + "_X_raw.npy", np.array(X, dtype=object))
    np.save(save_prefix + "_y_raw.npy", np.array(y))

    print("\nSaved:", save_prefix + "_X_raw.npy")
    print("Saved:", save_prefix + "_y_raw.npy")



if __name__ == "__main__":
    raw_dir = "./asl_alphabet/asl_alphabet_train"  
    save_prefix = "./data/processed/asl"
    process_dataset(raw_dir, save_prefix, limit=100)
