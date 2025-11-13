import numpy as np
from pathlib import Path

TARGET_DIR = Path(__file__).resolve().parent
dataset_dir = TARGET_DIR / "mnist" / "dataset"
pred_file = dataset_dir / "prediction.txt"
test_pixels = dataset_dir / "mnist_test_pixels.txt"
test_labels = dataset_dir / "mnist_test_labels.txt"
weights_dir = TARGET_DIR / "mlp_weights"

def approx_sign(x: np.ndarray) -> np.ndarray:
    # Match torch.clamp(x, -1, 1)
    x = np.clip(x, -1.0, 1.0)
    c1, c3, c5, c7, c9, c11 = (
        8.82341343192733,
        -86.6415008377027,
        388.964712077092,
        -797.090149675776,
        746.781707684981,
        -260.03867215588,
    )
    return (
        c1 * x
        + c3 * np.power(x, 3)
        + c5 * np.power(x, 5)
        + c7 * np.power(x, 7)
        + c9 * np.power(x, 9)
        + c11 * np.power(x, 11)
    )

def approx_relu(x: np.ndarray) -> np.ndarray:
    return 0.5 * (x * approx_sign(x) + x)

def test_mlp(W1, W2, pixels_file, labels_file, predictions_file=pred_file):
    X = np.loadtxt(pixels_file, dtype=np.float64)
    y_true = np.loadtxt(labels_file, dtype=int)
    if X.max() > 1.5:
        X = X / 255.0

    # pad all at once
    n = X.shape[0]
    X_pad = np.zeros((n, 1024), dtype=np.float64)
    X_pad[:, : X.shape[1]] = X

    # hidden layer
    H = X_pad @ W1
    H = approx_relu(H)               # vectorized
    Y = H @ W2                       # (n, 10)
    preds = np.argmax(Y[:, :10], axis=1)

    np.savetxt(predictions_file, preds, fmt="%d")
    acc = (preds == y_true).mean() * 100
    print(f"Final Accuracy: {acc:.2f}% ({(preds == y_true).sum()}/{len(y_true)})")
    return acc

if __name__ == "__main__":
    W1 = np.load(weights_dir / "fc1.npy")
    W2 = np.load(weights_dir / "fc2.npy")
    print("[INFO] Running MLP test on exported MNIST data...")
    test_mlp(W1, W2, test_pixels, test_labels)