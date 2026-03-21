project(add_heongpu LANGUAGES CXX CUDA)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

#  CUDA
find_package(CUDAToolkit REQUIRED)

#  HEonGPU (build from source)
set(HEONGPU_ROOT $ENV{HOME}/HEonGPU)
set(HEONGPU_BUILD ${HEONGPU_ROOT}/build)

# Add HEonGPU as a subdirectory to be built with the project
add_subdirectory(${HEONGPU_ROOT} ${HEONGPU_BUILD})

#  Project includes
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

#  Key generation
add_executable(client_key_generation src/client_key_generation_heongpu.cu) 

#  Server model preprocessing
add_executable(server_preprocess_model src/server_preprocess_model_heongpu.cu)

#  Client input preprocessing
add_executable(client_preprocess_input src/client_preprocess_input_heongpu.cu)

#  Client input encryption
add_executable(client_encode_encrypt_input 
    src/client_encode_encrypt_input_heongpu.cu 
    src/mlp_encryption_utils_heongpu.cu
)

#  Server encrypted computation
add_executable(server_encrypted_compute
    src/server_encrypted_compute_heongpu.cu
    src/mlp_encryption_utils_heongpu.cu
    src/mlp_heongpu.cu
)

#  Client decrypt + decode
add_executable(client_decrypt_decode
    src/client_decrypt_decode_heongpu.cu
    src/mlp_encryption_utils_heongpu.cu
)

#  Client postprocess
add_executable(client_postprocess src/client_postprocess_heongpu.cu)

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
    target_link_libraries(${EXE} PRIVATE heongpu CUDA::cudart OpenMP::OpenMP_CXX)
    set_target_properties(${EXE} PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endforeach()
