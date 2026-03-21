// Copyright (c) 2025 HomomorphicEncryption.org
// Apache v2 License

#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "params.h"           
#include "utils.h"            

#include "mlp_encryption_utils.h"        


int main(int argc, char* argv[]) {
    if (argc < 2 || !std::isdigit(argv[1][0])) {
        std::cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
        std::cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }

    auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
    const auto exe_path = fs::canonical(fs::path(argv[0])).parent_path();
    const auto submission_root = exe_path.parent_path();
    const auto repo_root = submission_root.parent_path();
    InstanceParams prms(size, repo_root);

    const bool countOnly = (argc >= 3 && std::string(argv[2]) == "--count_only");

    // Device select (single GPU path)
    cudaSetDevice(0);

    // Load HE artifacts
    auto context = read_context(prms);
    auto publicKey = read_public_key(prms);

    // Dataset
    std::vector<Sample> dataset;
    load_dataset(dataset, prms.test_input_file().c_str());
    if (dataset.empty()) {
        throw std::runtime_error("No data found in " + prms.test_input_file().string());
    }
    if (dataset.size() != prms.getBatchSize()) {
        throw std::runtime_error("Dataset size does not match instance size");
    }

    heongpu::HEEncoder<Scheme>   encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, publicKey);

    const double scale = std::pow(2.0, 30);

    fs::create_directories(prms.ctxtupdir());

    const size_t DISTINCT_PACKING = 4;
    const size_t REPEATS = 2;

    for (size_t i = 0; i < dataset.size(); i += DISTINCT_PACKING) {
        size_t slot_count = context->get_poly_modulus_degree() / 2;
        std::vector<double> vec;
        vec.reserve(slot_count);

        for (size_t k = 0; k < DISTINCT_PACKING; ++k) {
            size_t sample_idx = i + k;
            if (sample_idx < dataset.size()) {
                const float* in_f = dataset[sample_idx].image;
                for (size_t r = 0; r < REPEATS; ++r) {
                    for (size_t j = 0; j < static_cast<size_t>(NORMALIZED_DIM); ++j) {
                        vec.push_back(static_cast<double>(in_f[j]));
                    }
                }
            } else {
                for (size_t r = 0; r < REPEATS; ++r) {
                    for (size_t j = 0; j < static_cast<size_t>(NORMALIZED_DIM); ++j) {
                        vec.push_back(0.0);
                    }
                }
            }
        }

        heongpu::Plaintext<Scheme>  plain(context);
        encoder.encode(plain, vec, scale);

        heongpu::Ciphertext<Scheme> ctxt(context);
        encryptor.encrypt(ctxt, plain);

        if (!countOnly) {
            auto outPath = prms.ctxtupdir() / ("cipher_input_" + std::to_string(i / DISTINCT_PACKING) + ".bin");
            heongpu::serializer::save_to_file(ctxt, outPath);
        }
    }

    return 0;
}
