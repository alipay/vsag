
add_subdirectory (simd)
set (SIMD_QUANT_SRCS quantization/sq4_simd.cpp quantization/sq8_simd.cpp quantization/fp32_simd.cpp quantization/sq4_uniform_simd.cpp)
set_source_files_properties (${SIMD_QUANT_SRCS} PROPERTIES COMPILE_FLAGS "-msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2")
set_source_files_properties (${SIMD_QUANT_SRCS} PROPERTIES COMPILE_FLAGS "-mavx")
set_source_files_properties (${SIMD_QUANT_SRCS} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
set_source_files_properties (${SIMD_QUANT_SRCS} PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512pf -mavx512er -mavx512cd -mavx512vl -mavx512bw -mavx512dq -mavx512ifma -mavx512vbmi")


file (GLOB CPP_SRCS "*.cpp")
file (GLOB CPP_FACTORY_SRCS "factory/*.cpp")
file (GLOB CPP_CONJUGATE_GRAPH_SRCS "impl/*.cpp")
file (GLOB CPP_INDEX_SRCS "index/*.cpp")
file (GLOB CPP_HNSWLIB_SRCS "algorithm/hnswlib/*.cpp")
file (GLOB CPP_QUANTIZATION_SRCS "quantization/*.cpp")
list (FILTER CPP_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER CPP_FACTORY_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER CPP_CONJUGATE_GRAPH_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER CPP_INDEX_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER CPP_QUANTIZATION_SRCS EXCLUDE REGEX "_test.cpp")

set (VSAG_SRCS ${CPP_SRCS} ${CPP_FACTORY_SRCS} ${CPP_INDEX_SRCS}
        ${CPP_CONJUGATE_GRAPH_SRCS} ${CPP_HNSWLIB_SRCS} ${CPP_QUANTIZATION_SRCS})
add_library (vsag SHARED ${VSAG_SRCS})
add_library (vsag_static STATIC ${VSAG_SRCS})

set (VSAG_DEP_LIBS diskann pthread m dl simd fmt::fmt-header-only nlohmann_json::nlohmann_json roaring)
target_link_libraries (vsag ${VSAG_DEP_LIBS})
target_link_libraries (vsag_static ${VSAG_DEP_LIBS})
maybe_add_dependencies (vsag spdlog roaring openblas boost mkl)
maybe_add_dependencies (vsag_static spdlog roaring openblas boost mkl)
