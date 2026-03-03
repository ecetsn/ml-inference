#include "mlp_heongpu.h"

#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
constexpr double kDefaultScale = static_cast<double>(1ULL << 30);
constexpr double kPolyInputScale = static_cast<double>(1ULL << 40);
}

heongpu::Ciphertext<Scheme> dense_matvec_naive(
    heongpu::HEContext<Scheme>& he,
    heongpu::Ciphertext<Scheme>& x_ct,
    const DenseWeights& W,
    heongpu::Publickey<Scheme>& pk,
    heongpu::HEArithmeticOperator<Scheme>& op,
    heongpu::Galoiskey<Scheme>& rk,
    heongpu::HEEncoder<Scheme>& enc,
    heongpu::Relinkey<Scheme>& mk,
    int& depth) {
    const size_t slot_count = he->get_poly_modulus_degree() / 2;
    const size_t in_dim = W.in_dim;
    const size_t out_dim = W.out_dim;

    if (W.data.size() != in_dim * out_dim) {
        throw std::invalid_argument("dense_matvec_naive: DenseWeights must store in_dim*out_dim values (diagonalized layout expected)");
    }
    
    // Debug: Check weight range and first few values
    double w_min = W.data[0], w_max = W.data[0];
    for (double d : W.data) {
        w_min = std::min(w_min, d);
        w_max = std::max(w_max, d);
    }
    std::cerr << "[debug] DenseMatVec weights: range=[" << w_min << ", " << w_max << "], first5=[";
    for (int i=0; i<5 && i<W.data.size(); ++i) std::cerr << W.data[i] << (i==4?"":", ");
    std::cerr << "]" << std::endl;
    if (out_dim > slot_count) {
        throw std::invalid_argument("dense_matvec_naive: out_dim exceeds slot count; weights must be packed as diagonals");
    }

    constexpr int kBlockSize = 32;
    const int giant_steps = static_cast<int>((in_dim + kBlockSize - 1) / kBlockSize);

    heongpu::Ciphertext<Scheme> y_ct(he);
    bool first_term = true;

    std::vector<heongpu::Ciphertext<Scheme>> x_block(kBlockSize, he);
    x_block[0] = x_ct;
    for (int b = 1; b < kBlockSize; ++b) {
        op.rotate_rows(x_ct, x_block[b], rk, b);
    }

    for (int b = 0; b < kBlockSize; ++b) {
        bool block_has_terms = false;
        heongpu::Ciphertext<Scheme> sum_b(he);

        for (int j = 0; j < giant_steps; ++j) {
            const size_t idx = static_cast<size_t>(j) * kBlockSize + static_cast<size_t>(b);
            if (idx >= in_dim) {
                break;
            }

            heongpu::Ciphertext<Scheme> x_rot(he);
            if (j == 0) {
                x_rot = x_block[b];
            } else {
                op.rotate_rows(x_block[b], x_rot, rk, j * kBlockSize);
            }

            std::vector<double> row_plain(slot_count, 0.0);
            const double* prow = &W.data[idx * out_dim];
            std::copy(prow, prow + out_dim, row_plain.begin());

            heongpu::Plaintext<Scheme> pt_row(he);
            enc.encode(pt_row, row_plain, kDefaultScale);
            for (int d = 0; d < depth; ++d) {
                try {
                    op.mod_drop_inplace(pt_row);
                } catch (const std::exception& e) {
                    std::cerr << "[debug] mod_drop failure in dense_matvec_naive (idx="
                              << idx << ", depth=" << depth << "): "
                              << e.what() << std::endl;
                    throw;
                }
            }

            heongpu::Ciphertext<Scheme> prod(he);
            op.multiply_plain(x_rot, pt_row, prod);

            if (!block_has_terms) {
                sum_b = prod;
                block_has_terms = true;
            } else {
                op.add_inplace(sum_b, prod);
            }
        }

        if (block_has_terms) {
            if (first_term) {
                y_ct = sum_b;
                first_term = false;
            } else {
                op.add_inplace(y_ct, sum_b);
            }
        }
    }

    if (!first_term) {
        try {
            op.rescale_inplace(y_ct);
            ++depth;
        } catch (const std::exception& e) {
            std::cerr << "[debug] rescale failure in dense_matvec_naive (final): "
                      << e.what() << std::endl;
            throw;
        }
    }
    return y_ct;
}

heongpu::Ciphertext<Scheme> approx_relu_ct(
    heongpu::HEContext<Scheme>& he,
    heongpu::Ciphertext<Scheme>& x,
    heongpu::HEArithmeticOperator<Scheme>& op,
    heongpu::HEEncoder<Scheme>& enc,
    heongpu::Relinkey<Scheme>& mk,
    int& depth) {
    const size_t slot_count = he->get_poly_modulus_degree() / 2;
    const double c1  =  8.82341343192733;
    const double c3  = -86.6415008377027;
    const double c5  =  388.964712077092;
    const double c7  = -797.090149675776;
    const double c9  =  746.781707684981;
    const double c11 = -260.03867215588;

auto safe_rescale = [&](heongpu::Ciphertext<Scheme>& ct,
                        int* level,
                        const char* ctx) {
    std::cerr << "[debug] before rescale in " << ctx
              << " depth=" << ct.depth()
              << " level=" << (level ? *level : -1) << std::endl;
    try {
        op.rescale_inplace(ct);
        if (level) {
            ++(*level);
        }
        std::cerr << "[debug] after rescale in " << ctx
                  << " depth=" << ct.depth()
                  << " level=" << (level ? *level : -1) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[debug] RESCALE FAILURE in " << ctx
                  << ": " << e.what() << std::endl;
        throw;
    }
};

    auto drop_plain_levels = [&](heongpu::Plaintext<Scheme>& pt,
                                 int target_level,
                                 const char* ctx) {
        for (int i = 0; i < target_level; ++i) {
            try {
                op.mod_drop_inplace(pt);
            } catch (const std::exception& e) {
                std::cerr << "[debug] mod_drop failure in " << ctx
                          << " (drop " << (i + 1) << "/" << target_level
                          << "): " << e.what() << std::endl;
                throw;
            }
        }
    };

    auto encode_constant_plain = [&](double coeff,
                                     int target_level,
                                     heongpu::Ciphertext<Scheme>& ref_ct,
                                     const char* ctx) {
        std::vector<double> vec(slot_count, coeff);
        heongpu::Plaintext<Scheme> pt(he);
        enc.encode(pt, vec, ref_ct.scale());
        drop_plain_levels(pt, target_level, ctx);
        return pt;
    };

    int lvl_x = depth;
    heongpu::Ciphertext<Scheme> u(he);
    op.multiply(x, x, u);
    op.relinearize_inplace(u, mk);
    int lvl_u = depth;
    safe_rescale(u, &lvl_u, "approx_relu_ct u=x^2");

    auto promote_cipher_level = [&](heongpu::Ciphertext<Scheme>& ct,
                                    int& lvl,
                                    int target_level,
                                    const char* ctx) {
        while (lvl < target_level) {
            try {
                op.mod_drop_inplace(ct);
                ++lvl;
            } catch (const std::exception& e) {
                std::cerr << "[debug] mod_drop failure in " << ctx
                          << ": " << e.what() << std::endl;
                throw;
            }
        }
    };
    /// it get past here and doesnt work after c11 
    heongpu::Ciphertext<Scheme> acc(he);
    auto pt_c11 = encode_constant_plain(c11, lvl_u, u, "approx_relu_ct c11");
    op.multiply_plain(u, pt_c11, acc);
    int lvl_acc = lvl_u;
    safe_rescale(acc, &lvl_acc, "approx_relu_ct c11*u");
    promote_cipher_level(u, lvl_u, lvl_acc, "approx_relu_ct promote u post c11");

    auto horner_mult = [&](double coeff,
                           const char* ctx) {
        auto pt_coeff = encode_constant_plain(coeff, lvl_acc, acc, ctx);
        op.add_plain_inplace(acc, pt_coeff);
        promote_cipher_level(u, lvl_u, lvl_acc, "approx_relu_ct promote u pre Horner");
        heongpu::Ciphertext<Scheme> tmp(he);
        op.multiply(acc, u, tmp);
        op.relinearize_inplace(tmp, mk);
        acc = std::move(tmp);
        safe_rescale(acc, &lvl_acc, "approx_relu_ct Horner");
        promote_cipher_level(u, lvl_u, lvl_acc, "approx_relu_ct promote u post Horner");
    };

    horner_mult(c9, "approx_relu_ct c9");
    horner_mult(c7, "approx_relu_ct c7");
    horner_mult(c5, "approx_relu_ct c5");
    horner_mult(c3, "approx_relu_ct c3");
    auto pt_c1 = encode_constant_plain(c1, lvl_acc, acc, "approx_relu_ct c1");
    op.add_plain_inplace(acc, pt_c1);

    heongpu::Ciphertext<Scheme> sign_ct(he);
    promote_cipher_level(x, lvl_x, lvl_acc, "approx_relu_ct promote x before sign");
    op.multiply(x, acc, sign_ct);
    op.relinearize_inplace(sign_ct, mk);
    safe_rescale(sign_ct, &lvl_acc, "approx_relu_ct sign");
    promote_cipher_level(x, lvl_x, lvl_acc, "approx_relu_ct promote x after sign");

    auto pt_one = encode_constant_plain(1.0, lvl_acc, sign_ct, "approx_relu_ct one");
    op.add_plain_inplace(sign_ct, pt_one);

    heongpu::Ciphertext<Scheme> relu_mul(he);
    promote_cipher_level(x, lvl_x, lvl_acc, "approx_relu_ct promote x before relu");
    op.multiply(x, sign_ct, relu_mul);
    op.relinearize_inplace(relu_mul, mk);
    safe_rescale(relu_mul, &lvl_acc, "approx_relu_ct x*(sign+1)");

    auto pt_half = encode_constant_plain(0.5, lvl_acc, relu_mul, "approx_relu_ct half");
    heongpu::Ciphertext<Scheme> relu_ct(he);
    op.multiply_plain(relu_mul, pt_half, relu_ct);
    
    // HEonGPU requires rescale before rotation/next op if rescale_required is true.
    // The previous multiply_plain set rescale_required = true.
    op.rescale_inplace(relu_ct);
    depth = lvl_acc + 1;
    return relu_ct;
}

heongpu::Ciphertext<Scheme> mlp_heongpu(
    heongpu::HEContext<Scheme>& he,
    heongpu::Ciphertext<Scheme>& x_ct,
    const DenseWeights& W1,
    const DenseWeights& W2,
    heongpu::Publickey<Scheme>& pk,
    heongpu::HEArithmeticOperator<Scheme>& op,
    heongpu::Galoiskey<Scheme>& rk,
    heongpu::HEEncoder<Scheme>& enc,
    heongpu::Relinkey<Scheme>& mk) {
    int depth = 0;
    auto h_ct = dense_matvec_naive(he, x_ct, W1, pk, op, rk, enc, mk, depth);
    std::cerr << "[debug] Before activation depth=" << depth << std::endl;
    h_ct = approx_relu_ct(he, h_ct, op, enc, mk, depth);
    std::cerr << "[debug] After activation depth=" << depth << std::endl;
    auto y_ct = dense_matvec_naive(he, h_ct, W2, pk, op, rk, enc, mk, depth);
    return y_ct;
}
