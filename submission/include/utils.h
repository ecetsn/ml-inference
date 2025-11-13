#pragma once
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "heongpu.cuh"

namespace fs = std::filesystem;

constexpr auto Scheme = heongpu::Scheme::CKKS;
