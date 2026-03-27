#pragma once

#include <cstddef>
#include <vector>
#include <string>

#include "params.cuh"           // contains Scheme typedef
#include <heongpu/heongpu.hpp>
#include "utils.cuh"
#include "mlp_encryption_utils.cuh"   // for DenseWeights + weight loader

// Pre-encode weights to avoid CPU bottleneck during inference
void pre_encode_weights(
    heongpu::HEContext<Scheme>& he,
    DenseWeights& W,
    heongpu::HEEncoder<Scheme>& enc,
    heongpu::HEArithmeticOperator<Scheme>& op,
    int depth);

// Naive dense matvec: y = x * W   (x is encrypted)
heongpu::Ciphertext<Scheme> dense_matvec_naive(
    heongpu::HEContext<Scheme>& he,
    heongpu::Ciphertext<Scheme>& x_ct,
    const DenseWeights& W,
    heongpu::Publickey<Scheme>& pk,
    heongpu::HEArithmeticOperator<Scheme>& op,
    heongpu::Galoiskey<Scheme>& rk,
    heongpu::HEEncoder<Scheme>& enc,
    heongpu::Relinkey<Scheme>& mk,
    int& depth);

// Approximate ReLU using polynomial
heongpu::Ciphertext<Scheme> approx_relu_ct(
    heongpu::HEContext<Scheme>& he,
    heongpu::Ciphertext<Scheme>& x,
    heongpu::HEArithmeticOperator<Scheme>& op,
    heongpu::HEEncoder<Scheme>& enc,
    heongpu::Relinkey<Scheme>& mk,
    int& depth);

// Full 2-layer MLP (FC1 → poly-ReLU → FC2)
heongpu::Ciphertext<Scheme> mlp_heongpu(
    heongpu::HEContext<Scheme>& he,
    heongpu::Ciphertext<Scheme>& x_ct,
    const DenseWeights& W1,
    const DenseWeights& W2,
    heongpu::Publickey<Scheme>& pk,
    heongpu::HEArithmeticOperator<Scheme>& op,
    heongpu::Galoiskey<Scheme>& rk,
    heongpu::HEEncoder<Scheme>& enc,
    heongpu::Relinkey<Scheme>& mk);
