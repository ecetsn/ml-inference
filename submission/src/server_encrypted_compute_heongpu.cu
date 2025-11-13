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

int main(int argc, char* argv[]) {

    if (argc < 2 || !std::isdigit(argv[1][0])) {
        std::cout << "Usage: " << argv[0] << " instance-size\n";
        std::cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }

    auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
    InstanceParams prms(size);

    // Initialize GPU and load HEonGPU context and keys
    cudaSetDevice(0);
    std::cout << "         [server] Loading HEonGPU context and keys..." << std::endl;

    auto context = heongpu::serializer::load_from_file<heongpu::HEContext<Scheme>>(prms.pubkeydir() / "cc.bin");
    auto public_key = heongpu::serializer::load_from_file<heongpu::Publickey<Scheme>>(prms.pubkeydir() / "pk.bin");
    auto eval_key = heongpu::serializer::load_from_file<heongpu::EvalKey<Scheme>>(prms.pubkeydir() / "ek.bin");

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEArithmeticOperator<Scheme> op(context, encoder);

    // Process encrypted inputs
    fs::create_directories(prms.ctxtdowndir());
    std::cout << "         [server] Running encrypted MLP inference..." << std::endl;

    for (size_t i = 0; i < prms.getBatchSize(); ++i) {
        auto input_ctxt_path = prms.ctxtupdir() / ("cipher_input_" + std::to_string(i) + ".bin");

        heongpu::Ciphertext<Scheme> ctxt_input;
        if (!heongpu::serializer::load_from_file(input_ctxt_path, ctxt_input)) {
            throw std::runtime_error("Failed to load ciphertext from " + input_ctxt_path.string());
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Perform encrypted inference
        // implement this MLP version using HEonGPU operators 
        auto ctxt_result = mlp_heongpu(context, op, ctxt_input);  

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "         [server] Execution time for ciphertext " << i 
                  << " : " << duration.count() << " seconds" << std::endl;

        auto result_ctxt_path = prms.ctxtdowndir() / ("cipher_result_" + std::to_string(i) + ".bin");
        heongpu::serializer::save_to_file(ctxt_result, result_ctxt_path);
    }

    std::cout << "         [server] All ciphertexts processed successfully." << std::endl;
    return 0;
}