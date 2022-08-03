cmake_minimum_required(VERSION 3.10)

project(  profile_training  )

# This sets the C++ version to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Import Torch C++ (search includes /usr/local/share/cmake)
find_package(Torch REQUIRED)

# Make target
add_executable(  profile_training   profile_training.cpp  )

# Link Torch C++ libs
target_link_libraries(  profile_training  "${TORCH_LIBRARIES}")