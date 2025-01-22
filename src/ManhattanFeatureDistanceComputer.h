#pragma once

#include "IFeatureDistanceComputer.h"
#include "FeatureVector.h"

namespace cbir {

template<typename dtype, size_t D>
class ManhattanFeatureDistanceComputer {
public:
    double compute(const FeatureVector<dtype, D>& v1, const FeatureVector<dtype, D>& v2) override;
    double min_distance() override;
    double max_distance() override;
};

} // namespace cbir
