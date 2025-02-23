cmake_minimum_required (VERSION 3.20)
project(cbir)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(GTest REQUIRED)

add_executable(cbir
                src/main.cpp
                src/FeatureVector.h
                src/HsvHistogramFeatureVectorComputer.hpp
                src/IFeatureDistanceComputer.h
                src/IFeatureStorage.h
                src/IFeatureVectorComputer.h
                src/JsonFeatureStorage.hpp
                src/ManhattanFeatureDistanceComputer.hpp)
target_link_libraries(cbir PUBLIC ${OpenCV_LIBS})

set_target_properties(cbir
                      PROPERTIES CXX_STANDARD 17 
                      CXX_STANDARD_REQUIRED ON)
target_compile_options(cbir PUBLIC -Wall -Wextra -Werror -pedantic)

add_executable(tests
                src/FeatureVector.h
                src/HsvHistogramFeatureVectorComputer.hpp
                src/IFeatureDistanceComputer.h
                src/IFeatureStorage.h
                src/IFeatureVectorComputer.h
                src/JsonFeatureStorage.hpp
                src/ManhattanFeatureDistanceComputer.hpp

                tests/tests_HsvHistogramFeatureVectorComputer.cpp
                tests/tests_ManhattanFeatureDistanceComputer.cpp
                tests/tests_JsonFeatureStorage.cpp
)
set_target_properties(tests
                      PROPERTIES CXX_STANDARD 17 
                      CXX_STANDARD_REQUIRED ON)
target_compile_options(tests PUBLIC -Wall -Wextra -Werror -pedantic)
target_include_directories(tests PRIVATE "src/" "tests/" ${GTEST_INCLUDE_DIRS})
target_link_libraries(tests ${OpenCV_LIBS} ${GTEST_BOTH_LIBRARIES})

enable_testing()
add_test(tests tests) 
