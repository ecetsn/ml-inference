// Copyright (c) 2025 HomomorphicEncryption.org
// All rights reserved.
// Licensed under Apache v2. See LICENSE.md.

#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "params.cuh"
#include "utils.cuh"

#include <heongpu/heongpu.hpp>
#include "mlp_encryption_utils.cuh"


int main(int argc, char* argv[]) {
  if (argc < 2 || !std::isdigit(argv[1][0])) {
        std::cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
        std::cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }

    const auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
    const auto exe_path = fs::canonical(fs::path(argv[0])).parent_path();
    const auto submission_root = exe_path.parent_path();
    // Harness expects IO, datasets, etc. under repo root (two levels above submissions/mlp/)
    const auto repo_root = submission_root.parent_path().parent_path();
    InstanceParams prms(size, repo_root);

    // Use it for memory pool
    cudaSetDevice(0);

    // Load context & secret key
    auto context    = read_context(prms);
    auto secret_key = read_secret_key(prms);

    // Output setup
    auto pred_path = prms.encrypted_model_predictions_file();
    std::ofstream out(pred_path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output predictions file: " + pred_path.string());
    }

    const size_t DISTINCT_PACKING = 4;
    const size_t num_batches = (prms.getBatchSize() + DISTINCT_PACKING - 1) / DISTINCT_PACKING;

    std::vector<heongpu::Ciphertext<Scheme>> result_batches;
    std::cout << "         [client] Loading result batch file..." << std::endl;
    load_batch(result_batches, prms.ctxtdowndir() / "cipher_result_batch.bin", context);

    if (result_batches.size() != num_batches) {
        throw std::runtime_error("Client: Decrypted batch size does not match expected size");
    }

    std::vector<int> all_predictions(prms.getBatchSize());

    std::cout << "         [client] Decrypting and decoding batches using OpenMP..." << std::endl;
    #pragma omp parallel
    {
        #pragma omp for
        for (size_t i = 0; i < num_batches; ++i) {
            // Each call to mlp_decrypt creates its own internal decryptor/encoder
            auto full_logits = mlp_decrypt(context, secret_key, result_batches[i]);
            constexpr int kNumClasses = 10;

            for (size_t k = 0; k < DISTINCT_PACKING; ++k) {
                size_t sample_idx = i * DISTINCT_PACKING + k;
                if (sample_idx < prms.getBatchSize()) {
                    // Each unique image k starts at slot (2*k * 1024)
                    float* sample_logits = &full_logits[2 * k * NORMALIZED_DIM];
                    all_predictions[sample_idx] = argmax(sample_logits, kNumClasses);
                }
            }
        }
    }

    for (int pred : all_predictions) {
        out << pred << '\n';
    }
    return 0;
}
