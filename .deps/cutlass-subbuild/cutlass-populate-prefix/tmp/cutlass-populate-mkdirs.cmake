# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/projects/bdjx/hxia3/vllm/.deps/cutlass-src")
  file(MAKE_DIRECTORY "/projects/bdjx/hxia3/vllm/.deps/cutlass-src")
endif()
file(MAKE_DIRECTORY
  "/projects/bdjx/hxia3/vllm/.deps/cutlass-build"
  "/projects/bdjx/hxia3/vllm/.deps/cutlass-subbuild/cutlass-populate-prefix"
  "/projects/bdjx/hxia3/vllm/.deps/cutlass-subbuild/cutlass-populate-prefix/tmp"
  "/projects/bdjx/hxia3/vllm/.deps/cutlass-subbuild/cutlass-populate-prefix/src/cutlass-populate-stamp"
  "/projects/bdjx/hxia3/vllm/.deps/cutlass-subbuild/cutlass-populate-prefix/src"
  "/projects/bdjx/hxia3/vllm/.deps/cutlass-subbuild/cutlass-populate-prefix/src/cutlass-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/projects/bdjx/hxia3/vllm/.deps/cutlass-subbuild/cutlass-populate-prefix/src/cutlass-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/projects/bdjx/hxia3/vllm/.deps/cutlass-subbuild/cutlass-populate-prefix/src/cutlass-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
