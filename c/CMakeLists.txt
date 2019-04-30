cmake_minimum_required (VERSION 2.8)
project (mnist_c_demo)
aux_source_directory(. DIR_SRCS)
add_executable(mnist_c_demo ${DIR_SRCS})
