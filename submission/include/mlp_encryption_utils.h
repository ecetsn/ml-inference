// Copyright (c) 2025
// Apache v2 License
#pragma once

#include <vector>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "utils.h"
#include "params.h"
#include "heongpu.cuh"

namespace fs = std::filesystem;

// Basic constants and dataset structure
constexpr int MNIST_DIM      = 784;
constexpr int NORMALIZED_DIM = 1024;

struct Sample {
    float image[NORMALIZED_DIM] {};
};

struct DenseWeights {
    std::size_t in_dim;      // input size
    std::size_t out_dim;     // output size
    // row-major: row i starts at i * out_dim
    std::vector<double> data;
};

// Function declarations (non-templated HE I/O)
heongpu::HEContext<Scheme> read_context(const InstanceParams& prms);
heongpu::Publickey<Scheme> read_public_key(const InstanceParams& prms);
heongpu::Secretkey<Scheme> read_secret_key(const InstanceParams& prms);
heongpu::Relinkey<Scheme>  read_relin_key(const InstanceParams& prms);
heongpu::Galoiskey<Scheme> read_galois_key(const InstanceParams& prms);

// MLP / HE helper functions
// NOTE: matches your .cpp: input is std::vector<float>
heongpu::Ciphertext<Scheme> mlp_encrypt(
    heongpu::HEContext<Scheme>&  ctx,
    heongpu::Publickey<Scheme>&  pk,
    const std::vector<float>&    input
);

// NOTE: matches your .cpp: no out_len parameter
std::vector<float> mlp_decrypt(
    heongpu::HEContext<Scheme>&  ctx,
    heongpu::Secretkey<Scheme>&  sk,
    heongpu::Ciphertext<Scheme>& ct
);

// Dataset utilities
void load_dataset(std::vector<Sample>& dataset, const char* filename);
int  argmax(float* A, int N);

// Fully-connected weight loading 
DenseWeights load_fc_weights_txt(
    const std::string& path,
    std::size_t        in_dim,
    std::size_t        out_dim
);
