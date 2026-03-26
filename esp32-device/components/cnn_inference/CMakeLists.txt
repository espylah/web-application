idf_component_register(
    SRCS "cnn_inference.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp_common json espressif__esp-dl
)

# Locate the esp-dl component directory (resolved after idf_component_register)
idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif ("___idf_esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()

if (NOT espdl_dir)
    message(FATAL_ERROR "cnn_inference: esp-dl component not found. Check idf_component.yml.")
endif()

set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

# Only embed the model when loading from rodata (not partition mode)
if (NOT CONFIG_CNN_MODEL_PARTITION)
    target_add_aligned_binary_data(${COMPONENT_LIB} "lettercnn.espdl" BINARY)
endif()
