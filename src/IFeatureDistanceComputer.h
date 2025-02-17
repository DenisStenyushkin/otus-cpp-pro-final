#pragma once

#include "FeatureVector.h"

namespace cbir {

template<typename dtype, size_t D>
class IFeatureDistanceComputer {
public:
    virtual double compute(const FeatureVector<dtype, D>& v1, const FeatureVector<dtype, D>& v2) const = 0;
};

} // namespace cbir
