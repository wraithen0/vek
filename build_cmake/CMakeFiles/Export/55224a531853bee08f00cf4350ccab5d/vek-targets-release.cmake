#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "vek::vek_static" for configuration "Release"
set_property(TARGET vek::vek_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(vek::vek_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libvek_static.a"
  )

list(APPEND _cmake_import_check_targets vek::vek_static )
list(APPEND _cmake_import_check_files_for_vek::vek_static "${_IMPORT_PREFIX}/lib/libvek_static.a" )

# Import target "vek::vek_shared" for configuration "Release"
set_property(TARGET vek::vek_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(vek::vek_shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libvek_shared.so"
  IMPORTED_SONAME_RELEASE "libvek_shared.so"
  )

list(APPEND _cmake_import_check_targets vek::vek_shared )
list(APPEND _cmake_import_check_files_for_vek::vek_shared "${_IMPORT_PREFIX}/lib/libvek_shared.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
