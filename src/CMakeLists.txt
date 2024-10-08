
add_subdirectory (simd)


file (GLOB CPP_SRCS "*.cpp")
file (GLOB CPP_FACTORY_SRCS "factory/*.cpp")
file (GLOB CPP_CONJUGATE_GRAPH_SRCS "impl/*.cpp")
file (GLOB CPP_INDEX_SRCS "index/*.cpp")
file (GLOB CPP_HNSWLIB_SRCS "algorithm/hnswlib/*.cpp")
file (GLOB CPP_QUANTIZATION_SRCS "quantization/*.cpp")
file (GLOB CPP_DATA_CELL_SRCS "data_cell/*.cpp")
list (FILTER CPP_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER CPP_FACTORY_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER CPP_CONJUGATE_GRAPH_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER CPP_INDEX_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER CPP_QUANTIZATION_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER CPP_DATA_CELL_SRCS EXCLUDE REGEX "_test.cpp")

set (VSAG_SRCS ${CPP_SRCS} ${CPP_FACTORY_SRCS} ${CPP_INDEX_SRCS}
        ${CPP_CONJUGATE_GRAPH_SRCS} ${CPP_HNSWLIB_SRCS} ${CPP_QUANTIZATION_SRCS} ${CPP_DATA_CELL_SRCS})
add_library (vsag SHARED ${VSAG_SRCS})
add_library (vsag_static STATIC ${VSAG_SRCS})

set (VSAG_DEP_LIBS diskann pthread m dl simd fmt::fmt-header-only nlohmann_json::nlohmann_json roaring)
target_link_libraries (vsag ${VSAG_DEP_LIBS})
target_link_libraries (vsag_static ${VSAG_DEP_LIBS})
maybe_add_dependencies (vsag spdlog roaring openblas boost mkl)
maybe_add_dependencies (vsag_static spdlog roaring openblas boost mkl)
