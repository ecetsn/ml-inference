// Copyright (c) 2025 HomomorphicEncryption.org
// All rights reserved.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================

#include "params.h"
#include "utils.h"
#include <heongpu/heongpu.hpp>
 

int main(int argc, char* argv[]) {

    if (argc < 2 || !std::isdigit(argv[1][0])) {
        std::cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
        std::cout << "  Instance-size: 0-TOY, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }
    auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
    const auto exe_path = fs::canonical(fs::path(argv[0])).parent_path();
    const auto submission_root = exe_path.parent_path();
    // Harness expects IO under repo root (one level above submission/)
    const auto repo_root = submission_root.parent_path();
    InstanceParams prms(size, repo_root);

    cudaSetDevice(0);

    // Initialize encryption parameters for the CKKS scheme
    heongpu::HEContext<Scheme> context = heongpu::GenHEContext<Scheme>();

    size_t poly_modulus_degree = 16384;
    context->set_poly_modulus_degree(poly_modulus_degree);
    // Extend the modulus chain so we have more rescale levels for deep MLP ops.
    context->set_coeff_modulus_bit_sizes(
        {40, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30},
        {60});
    context->generate();
    context->print_parameters();

    size_t in_dim = 1024;
    int B = 32;
    int T = static_cast<int>((in_dim + B - 1) / B);
    std::vector<int> rotations;
    rotations.reserve((B - 1) + (T - 1));

    // baby steps
    for (int b = 1; b < B; ++b) {
    rotations.push_back(b);
    }
    // giant steps
    for (int j = 1; j < T; ++j) {
    rotations.push_back(j * B);
    }
    std::sort(rotations.begin(), rotations.end());
    rotations.erase(std::unique(rotations.begin(), rotations.end()), rotations.end());
    

    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context);
    // Rotation keys needed according to the baby step giant step algorithm
    heongpu::Galoiskey<Scheme> galois_key(context, rotations);
    heongpu::Relinkey<Scheme> relin_key(context);

    keygen.generate_secret_key(secret_key);

    heongpu::Publickey<Scheme> public_key(context);
    
    keygen.generate_public_key(public_key, secret_key);
    keygen.generate_relin_key(relin_key, secret_key);
    keygen.generate_galois_key(galois_key, secret_key);

    fs::create_directories(prms.pubkeydir());
    fs::create_directories(prms.seckeydir());

    {
        std::ofstream ofs(prms.pubkeydir()/"cc.bin", std::ios::binary);
        context->save(ofs);
    }
    {
        std::ofstream ofs(prms.pubkeydir()/"pk.bin", std::ios::binary);
        public_key.save(ofs);
    }
    {
        std::ofstream ofs(prms.seckeydir()/"sk.bin", std::ios::binary);
        secret_key.save(ofs);
    }
    {
        std::ofstream ofs(prms.pubkeydir()/"rk.bin", std::ios::binary);
        galois_key.save(ofs);
    }
    {
        std::ofstream ofs(prms.pubkeydir()/"mk.bin", std::ios::binary);
        relin_key.save(ofs);
    }
    return 0;

}
