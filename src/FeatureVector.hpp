#pragma once

#include <array>
#include <cstddef>


namespace cbir {

template<typename dtype, size_t D>
using FeatureVector = std::array<dtype, D>;

} // namespace cbir
