#pragma once
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <heongpu/heongpu.hpp>

namespace fs = std::filesystem;

constexpr auto Scheme = heongpu::Scheme::CKKS;
