#include "mlp_encryption_utils.h"
#include "utils.h"
#include "heongpu.cuh"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace {
constexpr double kDefaultScale = static_cast<double>(1ULL << 30);
}

heongpu::HEContext<Scheme> read_context(const InstanceParams& prms) {
    const auto path = prms.pubkeydir() / "cc.bin";
    if (!fs::exists(path)) {
        throw std::runtime_error("read_context: missing " + path.string());
    }
    return heongpu::serializer::load_from_file<heongpu::HEContext<Scheme>>(path);
}

heongpu::Publickey<Scheme> read_public_key(const InstanceParams& prms) {
    const auto path = prms.pubkeydir() / "pk.bin";
    if (!fs::exists(path)) {
        throw std::runtime_error("read_public_key: missing " + path.string());
    }
    return heongpu::serializer::load_from_file<heongpu::Publickey<Scheme>>(path);
}

heongpu::Secretkey<Scheme> read_secret_key(const InstanceParams& prms) {
    const auto path = prms.seckeydir() / "sk.bin";
    if (!fs::exists(path)) {
        throw std::runtime_error("read_secret_key: missing " + path.string());
    }
    return heongpu::serializer::load_from_file<heongpu::Secretkey<Scheme>>(path);
}

heongpu::Relinkey<Scheme> read_relin_key(const InstanceParams& prms) {
    const auto path = prms.pubkeydir() / "mk.bin";
    if (!fs::exists(path)) {
        throw std::runtime_error("read_relin_key: missing " + path.string());
    }
    return heongpu::serializer::load_from_file<heongpu::Relinkey<Scheme>>(path);
}

heongpu::Galoiskey<Scheme> read_galois_key(const InstanceParams& prms) {
    const auto path = prms.pubkeydir() / "rk.bin";
    if (!fs::exists(path)) {
        throw std::runtime_error("read_galois_key: missing " + path.string());
    }
    return heongpu::serializer::load_from_file<heongpu::Galoiskey<Scheme>>(path);
}

heongpu::Ciphertext<Scheme> mlp_encrypt(heongpu::HEContext<Scheme>& ctx,
                                        heongpu::Publickey<Scheme>& pk,
                                        const std::vector<float>& input) {
    std::vector<double> input_d(input.begin(), input.end());
    if (input_d.empty()) {
        throw std::invalid_argument("mlp_encrypt: input vector must be non-empty");
    }

    const size_t slot_count = ctx.get_poly_modulus_degree() / 2;
    if (slot_count == 0) {
        throw std::runtime_error("mlp_encrypt: invalid slot count");
    }

    std::vector<double> packed(slot_count);
    for (size_t i = 0; i < slot_count; ++i) {
        packed[i] = input_d[i % input_d.size()];
    }

    heongpu::HEEncoder<Scheme> encoder(ctx);
    heongpu::HEEncryptor<Scheme> encryptor(ctx, pk);

    heongpu::Plaintext<Scheme> pt(ctx);
    encoder.encode(pt, packed, kDefaultScale);

    heongpu::Ciphertext<Scheme> ct(ctx);
    encryptor.encrypt(ct, pt);
    return ct;
}

std::vector<float> mlp_decrypt(heongpu::HEContext<Scheme>& ctx,
                               heongpu::Secretkey<Scheme>& sk,
                               heongpu::Ciphertext<Scheme>& ct) {
    heongpu::HEDecryptor<Scheme> decryptor(ctx, sk);
    heongpu::HEEncoder<Scheme> encoder(ctx);

    heongpu::Plaintext<Scheme> pt(ctx);
    decryptor.decrypt(pt, ct);

    std::vector<double> decoded;
    encoder.decode(decoded, pt);

    std::vector<float> result(decoded.size());
    std::transform(decoded.begin(), decoded.end(), result.begin(),
                   [](double v) { return static_cast<float>(v); });
    return result;
}

void load_dataset(std::vector<Sample>& dataset, const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("load_dataset: cannot open ") + filename);
    }

    Sample sample;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        for (int i = 0; i < MNIST_DIM; ++i) {
            iss >> sample.image[i];
        }
        for (int i = MNIST_DIM; i < NORMALIZED_DIM; ++i) {
            sample.image[i] = 0.0f;
        }
        dataset.push_back(sample);
    }
}

int argmax(float* values, int count) {
    if (count <= 0) {
        return -1;
    }
    int max_idx = 0;
    //std::cout<<values<<std::endl;
    for (int i = 1; i < count; ++i) {
        if (values[i] > values[max_idx]) {
            max_idx = i;
        }
    }
    return max_idx;
}

DenseWeights load_fc_weights_txt(const std::string& path,
                                 std::size_t in_dim,
                                 std::size_t out_dim) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        throw std::runtime_error("Cannot open weight file: " + path);
    }

    std::vector<double> values;
    values.reserve(in_dim * out_dim);

    double v = 0.0;
    while (fin >> v) {
        values.push_back(v);
    }

    if (values.size() != in_dim * out_dim) {
        throw std::runtime_error(
            "Weight size mismatch in " + path + " (expected " +
            std::to_string(in_dim * out_dim) + ", got " +
            std::to_string(values.size()) + ")");
    }

    DenseWeights W;
    W.in_dim = in_dim;
    W.out_dim = out_dim;
    W.data = std::move(values);
    return W;
}
