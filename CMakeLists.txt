cmake_minimum_required (VERSION 3.20)
project(cbir)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cbir src/main.cpp)
target_link_libraries(cbir PUBLIC ${OpenCV_LIBS})

set_target_properties(cbir
                      PROPERTIES CXX_STANDARD 17 
                      CXX_STANDARD_REQUIRED ON)
target_compile_options(cbir PUBLIC -Wall -Wextra -Werror -pedantic)
