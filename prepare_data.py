import numpy as np
from sklearn.model_selection import train_test_split

def prepare_dataset(X_path, y_path, save_prefix, test_size=0.2, val_size=0.2):
    X = np.load(X_path)
    y = np.load(y_path)

    # First split: train+val vs test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, stratify=y
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_ratio, shuffle=True, stratify=y_tv
    )

    np.save(save_prefix + "_X_train.npy", X_train)
    np.save(save_prefix + "_y_train.npy", y_train)
    np.save(save_prefix + "_X_val.npy", X_val)
    np.save(save_prefix + "_y_val.npy", y_val)
    np.save(save_prefix + "_X_test.npy", X_test)
    np.save(save_prefix + "_y_test.npy", y_test)

    print("Dataset prepared and saved")


if __name__ == "__main__":
    prepare_dataset(
        "./data/processed/asl_X_clean.npy",
        "./data/processed/asl_y_clean.npy",
        "./data/processed/asl"
    )

