import numpy as np

def clean_landmark_array(coords):
    """
    coords: shape (21, 3)
    returns numeric float array of shape (42,)
    """

    # Convert to float64
    coords = np.asarray(coords, dtype=np.float64)

    # remove z-dimension
    coords = coords[:, :2]

    # center around wrist
    wrist = coords[0]
    coords = coords - wrist

    # normalize
    max_val = np.max(np.abs(coords))
    if max_val == 0:
        max_val = 1.0
    coords = coords / max_val

    # flatten to numeric float64
    return coords.flatten().astype(np.float64)



def clean_and_save(raw_X_path, raw_y_path, save_prefix):
    X_raw = np.load(raw_X_path, allow_pickle=True)
    y_raw = np.load(raw_y_path, allow_pickle=True)

    X_valid = []
    y_valid = []

    for coords, label in zip(X_raw, y_raw):
        if coords.shape != (21, 3):
            continue  

        cleaned = clean_landmark_array(coords)
        if cleaned.shape != (42,):
            continue  

        X_valid.append(cleaned)
        y_valid.append(label)

    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)

    np.save(save_prefix + "_X_clean.npy", X_valid)
    np.save(save_prefix + "_y_clean.npy", y_valid)

    print("Saved:", len(X_valid), "valid cleaned samples")



if __name__ == "__main__":
    clean_and_save(
        "./data/processed/asl_X_raw.npy",
        "./data/processed/asl_y_raw.npy",
        "./data/processed/asl"
    )
