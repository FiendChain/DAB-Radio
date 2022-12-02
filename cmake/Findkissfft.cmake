cmake_minimum_required(VERSION 3.10)
project(kissfft)

set(KISSFFT_SRC_DIR ${CMAKE_SOURCE_DIR}/vendor/kissfft)
set(KISSFFT_FIXED_POINT 16)
set(KISSFFT_DATATYPE int16_t)
set(KISSFFT_OUTPUT_NAME kissfft)

add_library(kissfft SHARED
    ${KISSFFT_SRC_DIR}/kiss_fft.c
    ${KISSFFT_SRC_DIR}/kfc.c
    ${KISSFFT_SRC_DIR}/kiss_fftnd.c
    ${KISSFFT_SRC_DIR}/kiss_fftndr.c
    ${KISSFFT_SRC_DIR}/kiss_fftr.c)

target_include_directories(kissfft PUBLIC ${KISSFFT_SRC_DIR})
target_compile_definitions(kissfft PUBLIC 
    kiss_fft_scalar=${KISSFFT_DATATYPE}
    FIXED_POINT=${KISSFFT_FIXED_POINT}
    KISS_FFT_SHARED)
set_target_properties(kissfft PROPERTIES 
    DEFINE_SYMBOL KISS_FFT_BUILD
    C_VISIBILITY_PRESET hidden
    CXX_STANDARD 17)
