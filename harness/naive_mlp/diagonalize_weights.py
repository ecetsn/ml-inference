import numpy as np
from pathlib import Path

def diagonalize(matrix):
    """
    Transform an NxN matrix into its Halevi-Shoup diagonals.
    Mapping: d_k[i] = matrix[i, (i-k)%N]
    """
    N = matrix.shape[0]
    diagonals = np.zeros((N, N))
    for k in range(N):
        for i in range(N):
            # Diagonal k: v^{(k)}_i = W_{i, (i-k)%N}
            diagonals[k, i] = matrix[i, (i - k) % N]
    return diagonals

def main():
    weights_dir = Path("harness/naive_mlp/mlp_weights")
    output_dir = Path("submission/src/Mlp_Weights")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ["fc1", "fc2"]:
        npy_path = weights_dir / f"{name}.npy"
        if not npy_path.exists():
            print(f"[ERROR] {npy_path} not found.")
            continue
        
        print(f"[INFO] Processing {npy_path}...")
        W_loaded = np.load(npy_path)
        W_actual = W_loaded.T
        
        if W_actual.shape != (1024, 1024):
            print(f"[WARNING] Unexpected shape {W_actual.shape} for {name}")

        W_diag = diagonalize(W_actual)
        
        txt_path = output_dir / f"{name}.txt"
        print(f"[INFO] Saving diagonalized weights to {txt_path}...")
        np.savetxt(txt_path, W_diag, fmt="%.8f")

    print("[SUCCESS] Weight diagonalization complete.")

if __name__ == "__main__":
    main()
