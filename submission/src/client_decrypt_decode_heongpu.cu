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

#include "params.h"
#include "utils.h"

#include <heongpu/heongpu.hpp>
#include "mlp_encryption_utils.h"


int main(int argc, char* argv[]) {
  if (argc < 2 || !std::isdigit(argv[1][0])) {
        std::cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
        std::cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }

    const auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
    const auto exe_path = fs::canonical(fs::path(argv[0])).parent_path();
    const auto submission_root = exe_path.parent_path();
    const auto repo_root = submission_root.parent_path();
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

    // Batch decryption loop
    for (size_t i = 0; i < prms.getBatchSize(); ++i) {
        auto ctxt_path = prms.ctxtdowndir() / ("cipher_result_" + std::to_string(i) + ".bin");
        auto ctxt = heongpu::serializer::load_from_file<heongpu::Ciphertext<Scheme>>(ctxt_path);
        // taking the decrypted result happens in the client side ( with taking the argmax - logits)
        auto logits = mlp_decrypt(context, secret_key, ctxt);
        constexpr int kNumClasses = 10;
        //std::cout << "[debug] Logits for sample " << i << ": ";
        //for (int j = 0; j < kNumClasses; ++j) {
            //std::cout << std::fixed << std::setprecision(4) << logits[j] << (j == kNumClasses - 1 ? "" : ", ");
        //}
        //std::cout << std::endl;
        int pred = argmax(logits.data(), kNumClasses);
        out << pred << '\n';
    }

    std::cout << "[Info] Successfully wrote predictions to " << pred_path << "\n";
    return 0;
}
