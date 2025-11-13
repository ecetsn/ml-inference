/*
#include "heongpu.cuh"
#include "utils.h"

// assumes weights have been preloaded / serialized as plaintext diagonals

// skeleton of the mlp, just as an idea

heongpu::Ciphertext<Scheme> dense_matvec_naive(heongpu::HEContext<Scheme>& he, const heongpu::Ciphertext<Scheme>& x_ct, const DenseWeights& W) {
    // W.in_dim × W.out_dim, row-major
    auto y_ct = he.EncryptZero();
    for (size_t i = 0; i < W.in_dim; ++i) {
        // rotate encrypted vector by i to align feature x_i
        auto x_rot = (i == 0) ? x_ct : he.Rotate(x_ct, static_cast<int>(i));

        // encode i-th row of W as plaintext (placed in first out_dim slots)
        std::vector<double> row_plain(he.slot_count(), 0.0);
        const double* prow = &W.data[i * W.out_dim];
        std::copy(prow, prow + W.out_dim, row_plain.begin());
        auto pt_w = he.Encode(row_plain);

        // multiply and add
        auto prod = he.MulPlain(x_rot, pt_w);
        prod = he.Rescale(prod);
        y_ct = he.Add(y_ct, prod);
    }

    y_ct = he.Relinearize(y_ct);
    return y_ct;
}

heongpu::Ciphertext<Scheme> poly_sign_odd11(heongpu::HEContext<Scheme>& he, const heongpu::Ciphertext<Scheme>& x) {
    const double c1  =  8.82341343192733;
    const double c3  = -86.6415008377027;
    const double c5  =  388.964712077092;
    const double c7  = -797.090149675776;
    const double c9  =  746.781707684981;
    const double c11 = -260.03867215588;

    auto x2 = he.Mul(x, x); x2 = he.Rescale(x2);

    // Horner on even powers
    auto t = he.MulPlain(x2, he.EncodeScalar(c11)); t = he.Rescale(t);
    auto add_step = [&](heongpu::Ciphertext<Scheme> acc, double coeff) {
        acc = he.Mul(acc, x2); acc = he.Rescale(acc);
        return he.AddPlain(acc, he.EncodeScalar(coeff));
    };
    t = add_step(t, c9);
    t = add_step(t, c7);
    t = add_step(t, c5);
    t = add_step(t, c3);
    t = add_step(t, c1);

    t = he.Mul(t, x); t = he.Rescale(t);
    return t;
}

heongpu::Ciphertext<Scheme> approx_relu_ct(heongpu::HEContext<Scheme>& he, const heongpu::Ciphertext<Scheme>& x) {
    auto s = poly_sign_odd11(he, x);
    auto xs = he.Mul(x, s); xs = he.Rescale(xs);
    auto sum = he.Add(xs, x);
    return he.MulPlain(sum, he.EncodeScalar(0.5));
}

heongpu::Ciphertext<Scheme> mlp(heongpu::HEContext<Scheme> he, heongpu::Ciphertext<Scheme> x_ct) {

    // ---- First linear layer ----
    auto h_ct = dense_matvec_naive(he, x_ct, W1);  // encrypted matvec
    // ---- Polynomial ReLU ----
    h_ct = approx_relu_ct(he, h_ct);               // Horner polynomial
    // ---- Second linear layer ----
    auto y_ct = dense_matvec_naive(he, h_ct, W2);  // encrypted matvec
    return y_ct;                                   // logits in slots 0–9
}
*/