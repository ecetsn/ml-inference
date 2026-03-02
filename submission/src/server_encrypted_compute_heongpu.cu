// Copyright (c) 2025 HomomorphicEncryption.org
// All rights reserved.
// Licensed under the Apache v2 License.
// See the LICENSE.md file for details.

#include "params.h"
#include "utils.h"
#include "heongpu.cuh"
#include "mlp_encryption_utils.h"
#include <chrono>
#include <iostream>
#include <filesystem>
#include "mlp_heongpu.h"

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
    const auto repo_root = submission_root.parent_path();
    InstanceParams prms(size, repo_root);

    // Initialize GPU and load HEonGPU context and keys
    cudaSetDevice(0);
    std::cout << "         [server] Loading HEonGPU context,keys and weights..." << std::endl;

    auto context    = read_context(prms);
    auto public_key = read_public_key(prms);
    auto galois_key = read_galois_key(prms);
    auto relin_key  = read_relin_key(prms);
    auto weights_dir = prms.rtdir() / "src" / "Mlp_Weights";
    DenseWeights W_fc1 = load_fc_weights_txt((weights_dir / "fc1.txt").string(),
                                             FC1_IN_DIM, FC1_OUT_DIM);
    DenseWeights W_fc2 = load_fc_weights_txt((weights_dir / "fc2.txt").string(),
                                             FC2_IN_DIM, FC2_OUT_DIM);

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEArithmeticOperator<Scheme> op(context, encoder);

    std::cout << "W1: in_dim=" << W_fc1.in_dim << " out_dim=" << W_fc1.out_dim << std::endl;
    std::cout << "W2: in_dim=" << W_fc2.in_dim << " out_dim=" << W_fc2.out_dim << std::endl;


    // Process encrypted inputs
    fs::create_directories(prms.ctxtdowndir());
    std::cout << "         [server] Running encrypted MLP inference..." << std::endl;

    for (size_t i = 0; i < prms.getBatchSize(); ++i) {
        auto input_ctxt_path = prms.ctxtupdir() / ("cipher_input_" + std::to_string(i) + ".bin");
        auto ctxt_input = heongpu::serializer::load_from_file<heongpu::Ciphertext<Scheme>>(input_ctxt_path);

        auto start = std::chrono::high_resolution_clock::now();

        // Perform encrypted inference
        // implement this MLP version using HEonGPU operators 
        auto ctxt_result = mlp_heongpu(context, ctxt_input, W_fc1, W_fc2,
                                       public_key, op, galois_key, encoder, relin_key);

        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();
        std::cout << "         [server] Execution time for ciphertext " << i 
                  << " : " << duration << " seconds" << std::endl;

        auto result_ctxt_path = prms.ctxtdowndir() / ("cipher_result_" + std::to_string(i) + ".bin");
        heongpu::serializer::save_to_file(ctxt_result, result_ctxt_path);
    }

    std::cout << "         [server] All ciphertexts processed successfully." << std::endl;
    return 0;
}
