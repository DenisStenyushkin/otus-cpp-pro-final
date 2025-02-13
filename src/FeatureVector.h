#pragma once

#include <cstddef>
#include <initializer_list>


namespace cbir {

template<typename dtype, size_t D>
class FeatureVector {
public:
    FeatureVector();
    FeatureVector(std::initializer_list<dtype> values);

    dtype& operator[](size_t idx);
    size_t size();
};

} // namespace cbir
