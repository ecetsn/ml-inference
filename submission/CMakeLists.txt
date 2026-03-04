cmake_minimum_required(VERSION 3.23)
project(add_heongpu LANGUAGES CXX CUDA)

# --------------------------------------------------------------------
# 1. Language standards
# --------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# --------------------------------------------------------------------
# 2. CUDA
# --------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# --------------------------------------------------------------------
# 3. HEonGPU (build from source)
# --------------------------------------------------------------------
set(HEONGPU_ROOT $ENV{HOME}/HEonGPU)
set(HEONGPU_BUILD ${HEONGPU_ROOT}/build)

# Add HEonGPU as a subdirectory to be built with the project
add_subdirectory(${HEONGPU_ROOT} ${HEONGPU_BUILD})

# --------------------------------------------------------------------
# 4. Project includes
# --------------------------------------------------------------------
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# --------------------------------------------------------------------
# 5. Executables
# --------------------------------------------------------------------

# 1. Key generation
add_executable(client_key_generation src/client_key_generation_heongpu.cu) 

# 2. Server model preprocessing
add_executable(server_preprocess_model src/server_preprocess_model_heongpu.cu)

# 3. Client input preprocessing
add_executable(client_preprocess_input src/client_preprocess_input_heongpu.cu)

# 4. Client input encryption
add_executable(client_encode_encrypt_input 
    src/client_encode_encrypt_input_heongpu.cu 
    src/mlp_encryption_utils_heongpu.cu
)

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
    # Link against the 'heongpu' target provided by add_subdirectory
    target_link_libraries(${EXE} PRIVATE heongpu CUDA::cudart)
    set_target_properties(${EXE} PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endforeach()
