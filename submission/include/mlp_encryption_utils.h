// Copyright (c) 2025 HomomorphicEncryption.org
// Apache v2 License
#pragma once

#include <vector>
#include <filesystem>
#include <stdexcept>

#include "utils.h"
#include "params.h"


namespace fs = std::filesystem;

// Basic constants and dataset structure
constexpr int MNIST_DIM      = 784;
constexpr int NORMALIZED_DIM = 1024;

struct Sample {
    float image[NORMALIZED_DIM] {};
};

// Function declarations (non-templated HE I/O)
heongpu::HEContext<Scheme>    read_context(const InstanceParams& prms);
heongpu::Publickey<Scheme>  read_public_key(const InstanceParams& prms);
heongpu::Secretkey<Scheme>  read_secret_key(const InstanceParams& prms);

heongpu::Ciphertext<Scheme> mlp(const heongpu::HEContext<Scheme>& ctx, const heongpu::Ciphertext<Scheme>& input);
heongpu::Ciphertext<Scheme> mlp_encrypt(const heongpu::HEContext<Scheme>& ctx, const heongpu::Publickey<Scheme>& pk, const std::vector<double>& input);

std::vector<float> mlp_decrypt(const heongpu::HEContext<Scheme>& ctx, const heongpu::Secretkey<Scheme>& sk, const heongpu::Ciphertext<Scheme>& ct, size_t out_len);

void load_dataset(std::vector<Sample>& dataset, const char* filename);
int  argmax(float* A, int N);