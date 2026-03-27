// Copyright (c) 2025 HomomorphicEncryption.org
// All rights reserved.
// Licensed under the Apache v2 License.
// See the LICENSE.md file for details.

#include "params.cuh"
#include "utils.cuh"
#include <heongpu/heongpu.hpp>
#include "mlp_encryption_utils.cuh"
#include <chrono>
#include <iostream>
#include <filesystem>
#include "mlp_heongpu.cuh"

int main(int argc, char* argv[]) {

    if (argc < 2 || !std::isdigit(argv[1][0])) {
        std::cout << "Usage: " << argv[0] << " instance-size\n";
        std::cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }

    constexpr std::size_t FC1_IN_DIM   = 1024;
    constexpr std::size_t FC1_OUT_DIM  = 1024;
    constexpr std::size_t FC2_IN_DIM   = 1024;
    constexpr std::size_t FC2_OUT_DIM  = 1024;
    auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
    const auto exe_path = fs::canonical(fs::path(argv[0])).parent_path();
    const auto submission_root = exe_path.parent_path();
    // Harness expects IO, datasets, etc. under repo root (two levels above submissions/mlp/)
    const auto repo_root = submission_root.parent_path().parent_path();
    InstanceParams prms(size, repo_root);

    // Initialize GPU and load HEonGPU context and keys
    cudaSetDevice(0);
    std::cout << "         [server] Loading HEonGPU context,keys and weights..." << std::endl;

    auto context    = read_context(prms);
    auto public_key = read_public_key(prms);
    auto galois_key = read_galois_key(prms);
    auto relin_key  = read_relin_key(prms);
    auto weights_dir = prms.rtdir() / "submissions" / "mlp" / "src" / "Mlp_Weights";
    DenseWeights W_fc1 = load_fc_weights_txt((weights_dir / "fc1.txt").string(),
                                             FC1_IN_DIM, FC1_OUT_DIM);
    DenseWeights W_fc2 = load_fc_weights_txt((weights_dir / "fc2.txt").string(),
                                             FC2_IN_DIM, FC2_OUT_DIM);

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEArithmeticOperator<Scheme> op(context, encoder);

    //std::cout << "W1: in_dim=" << W_fc1.in_dim << " out_dim=" << W_fc1.out_dim << std::endl;
    //std::cout << "W2: in_dim=" << W_fc2.in_dim << " out_dim=" << W_fc2.out_dim << std::endl;

    // Pre-encode weights to avoid redundant CPU encoding in the inference loop
    std::cout << "         [server] Pre-encoding weights (CPU intensive)..." << std::endl;
    pre_encode_weights(context, W_fc1, encoder, op, 0);  // FC1 at level 0
    pre_encode_weights(context, W_fc2, encoder, op, 10); // FC2 at level 10


    // Process encrypted inputs
    fs::create_directories(prms.ctxtdowndir());
    std::cout << "         [server] Running encrypted MLP inference..." << std::endl;

    const size_t DISTINCT_PACKING = 4;
    const size_t num_batches = (prms.getBatchSize() + DISTINCT_PACKING - 1) / DISTINCT_PACKING;

    std::vector<heongpu::Ciphertext<Scheme>> input_batches;
    std::vector<heongpu::Ciphertext<Scheme>> result_batches;
    result_batches.reserve(num_batches);

    std::cout << "         [server] Loading all input batches from single file..." << std::endl;
    load_batch(input_batches, prms.ctxtupdir() / "cipher_input_batch.bin", context);

    if (input_batches.size() != num_batches) {
        throw std::runtime_error("Server: Loaded batch size does not match expected size");
    }

    for (size_t i = 0; i < num_batches; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        // Perform encrypted inference
        auto ctxt_result = mlp_heongpu(context, input_batches[i], W_fc1, W_fc2,
                                       public_key, op, galois_key, encoder, relin_key);

        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();
        if (i % 100 == 0 || i == num_batches - 1) {
            std::cout << "         [server] Execution time for batch " << i 
                      << " (up to " << DISTINCT_PACKING << " samples) : " << duration << " seconds" << std::endl;
        }

        result_batches.push_back(std::move(ctxt_result));
    }

    std::cout << "         [server] Saving all result batches to single file..." << std::endl;
    save_batch(result_batches, prms.ctxtdowndir() / "cipher_result_batch.bin");

    std::cout << "         [server] All ciphertexts processed successfully." << std::endl;
    return 0;
}
