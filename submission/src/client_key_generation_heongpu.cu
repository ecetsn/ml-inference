// Copyright (c) 2025 HomomorphicEncryption.org
// All rights reserved.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================

#include "params.h"
#include "utils.h"
#include "heongpu.cuh"


int main(int argc, char* argv[]) {

    if (argc < 2 || !std::isdigit(argv[1][0])) {
        std::cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
        std::cout << "  Instance-size: 0-TOY, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }
    auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
    InstanceParams prms(size);

    cudaSetDevice(0); // Use it for memory pool

    // Initialize encryption parameters for the CKKS scheme
    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);

    size_t poly_modulus_degree = 8192;
    context.set_poly_modulus_degree(poly_modulus_degree);
    context.set_coeff_modulus_bit_sizes({60, 30, 30, 30}, {60});
    context.generate();
    context.print_parameters();

    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    fs::create_directories(prms.pubkeydir());
    fs::create_directories(prms.seckeydir());

    heongpu::serializer::save_to_file(context, prms.pubkeydir()/"cc.bin");
    heongpu::serializer::save_to_file(public_key, prms.pubkeydir()/"pk.bin");
    heongpu::serializer::save_to_file(secret_key, prms.seckeydir()/"sk.bin");
    
    return 0;
}