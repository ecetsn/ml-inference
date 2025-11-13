import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np

TARGET_DIR = Path(__file__).resolve().parent
dataset_dir = TARGET_DIR / "mnist" / "dataset"
weights_dir = TARGET_DIR / "mlp_weights"
weights_dir.mkdir(parents=True, exist_ok=True)

train_pixels = dataset_dir / "mnist_train_pixels.txt"
train_labels = dataset_dir / "mnist_train_labels.txt"
test_pixels = dataset_dir / "mnist_test_pixels.txt"
test_labels = dataset_dir / "mnist_test_labels.txt"


class ApproxReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = torch.tensor([
            8.82341343192733,
            -86.6415008377027,
            388.964712077092,
            -797.090149675776,
            746.781707684981,
            -260.03867215588,
        ], dtype=torch.float32)

    def approx_sign(self, x):
        x = torch.clamp(x, -1.0, 1.0)
        c1, c3, c5, c7, c9, c11 = self.c
        return (c1*x
                + c3*x**3
                + c5*x**5
                + c7*x**7
                + c9*x**9
                + c11*x**11)

    def forward(self, x):
        return 0.5 * (x * self.approx_sign(x) + x)

# MLP: 1024×1024 fully connected layers
class PolyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 1024, bias=False)
        self.fc2 = nn.Linear(1024, 1024, bias=False)
        self.approx_relu = ApproxReLU()

    def forward(self, x):
        if x.size(1) < 1024:
            pad = torch.zeros(x.size(0), 1024 - x.size(1), device=x.device)
            x = torch.cat([x, pad], dim=1)
        h = self.approx_relu(self.fc1(x))
        y = self.fc2(h)
        return y[:, :10]

def load_mnist_from_txt(pixels_file, labels_file):
    X = np.loadtxt(pixels_file, dtype=np.float32)
    y = np.loadtxt(labels_file, dtype=np.int64)

    # Detect if normalization is needed
    if X.max() > 1.5:
        print("[INFO] Normalizing inputs (0–255 → 0–1)")
        X = X / 255.0

    # Convert to tensors
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    return X_tensor, y_tensor

def train_model(epochs=5, batch_size=128, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load training data
    X_train, y_train = load_mnist_from_txt(train_pixels, train_labels)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = PolyMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")

    model.eval()
    fc1 = model.fc1.weight.detach().cpu().numpy().T
    fc2 = model.fc2.weight.detach().cpu().numpy().T
    np.save(weights_dir / "fc1.npy", fc1)
    np.save(weights_dir / "fc2.npy", fc2)
    print(f"[INFO] Saved fc1.npy and fc2.npy in {weights_dir}")

    return model

@torch.no_grad()
def test_model(model, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X_test, y_test = load_mnist_from_txt(test_pixels, test_labels)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    acc = 100 * correct / total
    print(f"[INFO] Test accuracy: {acc:.2f}%")
    return acc

if __name__ == "__main__":
    model = train_model(epochs=5, lr=1e-3)
    test_model(model)