cmake_minimum_required(VERSION 3.14)
project(add_openfhe LANGUAGES CXX CUDA)

# --------------------------------------------------------------------
# 1.  Compiler / flags
# --------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# add_compile_options(-O3 -Wall -Wextra -Wno-unused-parameter)
option( BUILD_STATIC "Set to ON to include static versions of the library" OFF)

# --------------------------------------------------------------------
# 2.  Find the OpenFHE *package* that the install script put into
#     third_party/openfhe
# --------------------------------------------------------------------
#   scripts/get_openfhe.sh installs:
#     third_party/openfhe/lib/cmake/OpenFHE/OpenFHEConfig.cmake


# --------------------------------------------------------------------
# 3.  Link libraries
# --------------------------------------------------------------------



# set(CMAKE_SKIP_BUILD_RPATH FALSE)
# set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)
# set(CMAKE_BUILD_RPATH "${OpenFHE_LIBDIR}") 
# set(CMAKE_INSTALL_RPATH "${OpenFHE_LIBDIR}") 
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_link_options(-Wl,--no-as-needed)

# These should take care of setting the correct LD path without manually setting LD_LIBRARY_PATH. 
# But if a shared library or symbol lookup error occurs, you need to set the LD_LIBRARY_PATH environment 
# to the OpenFHE lib directory: export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/OpenFHE/lib


# --------------------------------------------------------------------
# 4.  Each *.cpp file becomes its own executable.
#     The eight stage names are hard-wired by the benchmark contract.
# --------------------------------------------------------------------
 

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 52 CACHE STRING "CUDA architectures" FORCE)
endif()


find_package(HEonGPU)

# 1. Key generation
add_executable(client_key_generation src/client_key_generation_heongpu.cu) 

# 2. Server model preprocessing
add_executable(server_preprocess_model src/server_preprocess_model_heongpu.cu)

# 3. Client input preprocessing
add_executable(client_preprocess_input src/client_preprocess_input_heongpu.cu)

# 4. Client input encryption -> need to implement mlp encryption utils with HEonGPU
add_executable(client_encode_encrypt_input src/client_encode_encrypt_input_heongpu.cu src/mlp_encryption_utils_heongpu.cu)

# 5. Server encrypted computation
add_executable(server_encrypted_compute
    src/server_encrypted_compute_heongpu.cu
    src/mlp_encryption_utils_heongpu.cu
    src/mlp_heongpu.cu
)

# 6. Client decrypt + decode
add_executable(client_decrypt_decode
    src/client_decrypt_decode_heongpu.cu
    src/mlp_encryption_utils_heongpu.cu
)

# 7. Client postprocess
add_executable(client_postprocess src/empty_stage.cu)

# List all your executables
set(MY_EXECUTABLES
    client_key_generation
    server_preprocess_model
    client_preprocess_input
    client_encode_encrypt_input
    server_encrypted_compute
    client_decrypt_decode
    client_postprocess
)

# Apply the same properties and libraries to each
foreach(EXE ${MY_EXECUTABLES})
    target_link_libraries(${EXE} PUBLIC HEonGPU::heongpu CUDA::cudart)
    set_target_properties(${EXE} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()
