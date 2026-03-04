import numpy as np
from pathlib import Path

def diagonalize(matrix):
    """
    Diagonalizes a weight matrix for HS MatVec with left rotations.
    y = x @ Matrix (x and y are row vectors)
    Mapping: D[k]_i = Matrix[(i + k) % N, i]
    where k is the diagonal index (and rotation amount), 
    and i is the slot index (and output index).
    """
    N = matrix.shape[0]
    diagonals = np.zeros((N, N))
    for k in range(N):
        for i in range(N):
            diagonals[k, i] = matrix[(i + k) % N, i]
    return diagonals

def main():
    weights_dir = Path("harness/naive_mlp/mlp_weights")
    output_dir = Path("submission/src/Mlp_Weights")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scaling factor for ReLU stability (pre-activations in [-1, 1])
    f = 0.1
    
    for name in ["fc1", "fc2"]:
        npy_path = weights_dir / f"{name}.npy"
        print(f"[INFO] Processing {npy_path}...")
        W_actual = np.load(npy_path)
        
        # Apply scaling
        if name == "fc1":
            W_final = W_actual * f
        else:
            W_final = W_actual * (1.0 / f)
            
        W_diag = diagonalize(W_final)
        
        txt_path = output_dir / f"{name}.txt"
        print(f"[INFO] Saving diagonalized weights to {txt_path} with scale factor...")
        np.savetxt(txt_path, W_diag, fmt="%.8f")

if __name__ == "__main__":
    main()
