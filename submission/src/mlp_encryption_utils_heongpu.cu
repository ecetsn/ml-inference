/*
// Copyright (c) 2025
// Apache v2 License

#include "mlp_encryption_utils.h"
#include "utils.h"
#include "heongpu.cuh"

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

heongpu::HEContext<Scheme> read_context(const InstanceParams& prms) {
    auto cc_path = prms.pubkeydir() / "cc.bin";
    if (!fs::exists(cc_path)) {
        throw std::runtime_error("read_context: missing " + cc_path.string());
    }
    return heongpu::serializer::load_from_file<heongpu::HEContext<Scheme>>(cc_path);
}

heongpu::Publickey<Scheme> read_public_key(const InstanceParams& prms) {
    auto pk_path = prms.pubkeydir() / "pk.bin";
    if (!fs::exists(pk_path)) {
        throw std::runtime_error("read_public_key: missing " + pk_path.string());
    }
    return heongpu::serializer::load_from_file<heongpu::Publickey<Scheme>>(pk_path);
}

heongpu::Secretkey<Scheme> read_secret_key(const InstanceParams& prms) {
    auto sk_path = prms.seckeydir() / "sk.bin";
    if (!fs::exists(sk_path)) {
        throw std::runtime_error("read_secret_key: missing " + sk_path.string());
    }
    return heongpu::serializer::load_from_file<heongpu::Secretkey<Scheme>>(sk_path);
}

// read_eval ? -> need to get the crypto context


heongpu::Ciphertext<Scheme> mlp_encrypt(const heongpu::HEContext<Scheme>& ctx, const heongpu::Publickey<Scheme>& pk, const std::vector<float>& input) {
    // Convert input to double (CKKS uses double precision)
    std::vector<double> input_d(input.begin(), input.end());

    // Determine number of slots (half ring dimension)
    const size_t slot_count = ctx.parameters().ring_dim() / 2;

    // Tile/pad vector to fill slot count
    std::vector<double> filled;
    filled.reserve(slot_count);
    for (size_t i = 0; i < slot_count; ++i)
        filled.push_back(input_d[i % input_d.size()]);

    heongpu::Encoder   encoder(ctx);
    heongpu::Encryptor encryptor(ctx, pk);

    heongpu::Plaintext<Scheme> pt(ctx);
    encoder.encode(pt, filled);  // automatic scale handled by context

    heongpu::Ciphertext<Scheme> ct(ctx);
    encryptor.encrypt(ct, pt);

    return ct;
}


std::vector<float> mlp_decrypt(const heongpu::HEContext<Scheme>& ctx, const heongpu::Secretkey<Scheme>& sk, const heongpu::Ciphertext<Scheme>& ct) {
    heongpu::Decryptor decryptor(ctx, sk);
    heongpu::Encoder   encoder(ctx);
 
    heongpu::Plaintext<Scheme> pt(ctx);
    decryptor.decrypt(pt, ct);

    std::vector<double> vals;
    encoder.decode(pt, vals);

    std::vector<float> result(vals.size());
    for (size_t i = 0; i < vals.size(); ++i)
        result[i] = static_cast<float>(vals[i]);

    return result;
}


void load_dataset(std::vector<Sample> &dataset, const char *filename) {
  std::ifstream file(filename);
  Sample sample;
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    // Read MNIST_DIM values from file
    for (int i = 0; i < MNIST_DIM; i++) {
      iss >> sample.image[i];
    }
    // Pad remaining values with 0.0 if NORMALIZED_DIM > MNIST_DIM
    for (int i = MNIST_DIM; i < NORMALIZED_DIM; i++) {
      sample.image[i] = 0.0f;
    }

    dataset.push_back(sample);
  }
}

int argmax(float *A, int N) {
  int max_idx = 0;
  for (int i = 1; i < N; i++) {
    if (A[i] > A[max_idx]) {
      max_idx = i;
    }
  }
  return max_idx;
}
*/