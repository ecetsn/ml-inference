# FHE Benchmarking Suite - ML Inference (HEonGPU)

This repository contains a high-performance, GPU-accelerated implementation of the ML-inference workload for the [HomomorphicEncryption.org](https://www.HomomorphicEncryption.org) FHE benchmarking suite. The harness currently supports MNIST model benchmarking as specified in the `harness/` directory.

## Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/ecetsn/ml-inference
    cd ml-inference
    ```

2.  **Prepare the environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Dependencies**:
    -   Requires **HEonGPU** library (typically located at `$HOME/HEonGPU`).
    -   Requires a CUDA-capable GPU.

## Running the Workload

The harness script `harness/run_submission.py` manages the entire pipeline, from building the submission to verifying the results.

```bash
python3 harness/run_submission.py [SIZE] [OPTIONS]
```

### Parameters
- **SIZE**: Instance size index:
  - `0`: **SINGLE** (1 sample)
  - `1`: **SMALL** (100 samples)
  - `2`: **MEDIUM** (1,000 samples)
  - `3`: **LARGE** (10,000 samples)
- **OPTIONS**:
  - `--num_runs [N]`: Number of benchmark iterations (default: 1).
  - `--seed [S]`: Random seed for reproducible input generation.
  - `--clrtxt 1`: Force rerun of the plaintext baseline computation.

### Example Run (Multi-Run Small Benchmark)
```console
$ python3 harness/run_submission.py 1 --seed 3 --num_runs 2
...
# --- INITIALIZATION (Performed Once) ---
17:13:34 [harness] 2: Client: Key Generation completed (elapsed: 250.07s)
17:13:34 [harness] 3: Server: Model preprocessing completed (elapsed: 0.01s)

# --- RUN 1 of 2 ---
17:13:38 [harness] 4: Harness: Input generation completed (elapsed: 3.59s)
         [client] Encrypting 100 samples into 25 batches (OpenMP)
17:13:47 [harness] 6: Client: Input encryption completed (elapsed: 8.47s)
         [server] Running encrypted MLP inference...
17:14:11 [harness] 7: Server: Inference computation completed (elapsed: 24.34s)
[harness] Encrypted model: 0.9400 (94/100 correct)
[total latency] 296.55s

# --- RUN 2 of 2 (Skipped Steps 2 & 3 as they are already active/cached) ---
17:14:17 [harness] 4: Harness: Input generation completed (elapsed: 2.65s)
17:14:27 [harness] 6: Client: Input encryption completed (elapsed: 9.74s)
17:14:51 [harness] 7: Server: Inference computation completed (elapsed: 24.24s)
[harness] Encrypted model: 0.9400 (94/100 correct)
[total latency] 297.65s
```

## HEonGPU Version Enhancements

Several engineering optimizations are implemented to improve throughput and latency:

- **Diagonalization**: Weights are pre-processed into a diagonal format. Matrix-vector multiplication is optimized using diagonal shifts instead of standard row-column access, significantly reducing the number of expensive rotation operations.
- **Slotting and Packing**: Multiple samples (4) are packed into a single ciphertext. This maximizes slot utilization and increases overall throughput.
- **Client-Side Parallelism**: Encryption, decryption, and decoding are parallelized using OpenMP to reduce client-side bottlenecks.
- **Weight Caching**: Model weights are encoded once during the preprocessing stage, eliminating redundant encoding overhead during inference runs.

## Training Harness (harness/naive_mlp)

The `harness/naive_mlp` directory contains tools for generating the model and weights used in this benchmark:

- **Weight Production**: Python scripts train a 1024x1024 MLP on MNIST and export the weights.
- **Polynomial Approximation**: Standard ReLU is non-linear and not directly supported in HE. An 11th-degree polynomial minimax approximation is used for the underlying sign function, resulting in a 12th-degree `ApproxReLU`. This activation function consumes **9 levels** of multiplicative depth using a sequential Horner-like evaluation.
- **Diagonalizing**: The `diagonalize_weights.py` script transforms standard weights into the diagonal format required by optimized GPU kernels.

## Description of Stages

The following stages are executed by the harness, mapping to specific binaries in the `submission/build` directory:

| Binary | Description |
|--------|-------------|
| `client_key_generation` | Generates all CKKS cryptographic keys and context at the client. |
| `server_preprocess_model` | Encodes and caches model weights using diagonalization. |
| `client_preprocess_input` | Performs initial plaintext data preparation. |
| `client_encode_encrypt_input` | Encodes and encrypts MNIST images into ciphertexts. |
| `server_encrypted_compute` | Performs homomorphic MLP inference over encrypted data on the GPU. |
| `client_decrypt_decode` | Decrypts and decodes the results at the client using OpenMP. |
| `client_postprocess` | Finalizes predictions and calculates labels from decrypted logits. |

## Security & Encryption Parameters

The **HEonGPU** library is configured with the following cryptographic parameters:

-   **Scheme**: CKKS
-   **Ring Dimension (N)**: 16,384.
-   **Security Level**: 128-bit security (HE-standard compliant).
-   **Modulus Chain**: 11 levels of 30-bit primes, providing sufficient depth for the 2-layer MLP (1 level per linear layer + 9 levels for `ApproxReLU`).

## Verification & Baseline

Correctness is ensured by comparing the encrypted model's output against a plaintext baseline:
- **Harness Model**: A standard Python/NumPy implementation of the same MLP architecture.
- **Accuracy Check**: The harness calculates accuracy for both versions, ensuring consistent results.

## Directory Structure

```
├─ README.md             # This file
├─ harness/              # Scripts to drive the workload implementation
│   ├─ run_submission.py # Main benchmark driver
│   ├─ naive_mlp/        # Training and weight production tools
│   └─ [...]
├─ submission/           # Core HEonGPU implementation
│   ├─ src/              # C++/CUDA source files
│   └─ include/          # Header files
├─ io/                   # Intermediate directory for client<->server communication
├─ measurements/         # Performance metrics and JSON logs
└─ datasets/             # Generated MNIST test datasets
```

---
