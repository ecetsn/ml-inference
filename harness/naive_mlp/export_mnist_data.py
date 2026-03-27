import torch
from torchvision import datasets, transforms
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "dataset"

def export_test_pixels_labels(data_dir=DATA_DIR, pixels_file="mnist_test_pixels.txt", labels_file="mnist_test_labels.txt", num_samples=-1, seed=None):
    """
    Export MNIST test dataset to separate label and pixel files.
    """
    if seed is not None:
        torch.manual_seed(seed)

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    total_samples = len(test_dataset)
    samples_to_export = total_samples if num_samples == -1 else min(num_samples, total_samples)

    pixels_path = data_dir / pixels_file
    labels_path = data_dir / labels_file

    print(f"[INFO] Exporting {samples_to_export} MNIST TEST samples to:")
    print(f"  Pixels: {pixels_path}")
    print(f"  Labels: {labels_path}")

    with open(labels_path, "w") as label_f, open(pixels_path, "w") as pixel_f:
        for i, (image, label) in enumerate(test_dataset):
            if i >= samples_to_export:
                break
            flattened = image.view(-1).numpy()
            label_f.write(f"{label}\n")
            pixel_values = " ".join(f"{v:.6f}" for v in flattened)
            pixel_f.write(f"{pixel_values}\n")

    print("[INFO] Test export completed successfully.")


def export_train_pixels_labels(data_dir=DATA_DIR, pixels_file="mnist_train_pixels.txt", labels_file="mnist_train_labels.txt", num_samples=-1, seed=None):
    """
    Export MNIST training dataset to separate label and pixel files.
    """
    if seed is not None:
        torch.manual_seed(seed)

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)

    total_samples = len(train_dataset)
    samples_to_export = total_samples if num_samples == -1 else min(num_samples, total_samples)

    pixels_path = data_dir / pixels_file
    labels_path = data_dir / labels_file

    print(f"[INFO] Exporting {samples_to_export} MNIST TRAIN samples to:")
    print(f"  Pixels: {pixels_path}")
    print(f"  Labels: {labels_path}")

    with open(labels_path, "w") as label_f, open(pixels_path, "w") as pixel_f:
        for i, (image, label) in enumerate(train_dataset):
            if i >= samples_to_export:
                break
            flattened = image.view(-1).numpy()
            label_f.write(f"{label}\n")
            pixel_values = " ".join(f"{v:.6f}" for v in flattened)
            pixel_f.write(f"{pixel_values}\n")

    print("[INFO] Train export completed successfully.")


def export_test_data(data_dir=DATA_DIR, output_file="mnist_test.txt", num_samples=-1, seed=None):
    base = Path(output_file).stem
    export_test_pixels_labels(
        data_dir=data_dir,
        pixels_file=f"{base}_pixels.txt",
        labels_file=f"{base}_labels.txt",
        num_samples=num_samples,
        seed=seed,
    )


def export_train_data(data_dir=DATA_DIR, output_file="mnist_train.txt", num_samples=-1, seed=None):
    base = Path(output_file).stem
    export_train_pixels_labels(
        data_dir=data_dir,
        pixels_file=f"{base}_pixels.txt",
        labels_file=f"{base}_labels.txt",
        num_samples=num_samples,
        seed=seed,
    )

if __name__ == "__main__":
    print("[INFO] Export mode: Loading and exporting MNIST data...")

    export_train_data(DATA_DIR, "mnist_train.txt", num_samples=-1)
    export_test_data(DATA_DIR, "mnist_test.txt", num_samples=-1)

    print("[INFO] All exports completed successfully.")
