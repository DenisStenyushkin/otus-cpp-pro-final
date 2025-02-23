#pragma once

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "IFeatureDistanceComputer.h"
#include "FeatureVector.h"

namespace cbir {

template<typename dtype, size_t D>
class ManhattanFeatureDistanceComputer : public IFeatureDistanceComputer<dtype, D> {
public:
    double compute(const FeatureVector<dtype, D>& v1, const FeatureVector<dtype, D>& v2) const override {
        std::vector<double> elem_distances(D);
        for (size_t i = 0; i < D; ++i) {
            elem_distances[i] = std::abs(v2[i] - v1[i]);
        }
        auto dist = std::accumulate(elem_distances.cbegin(), elem_distances.cend(), 0.0);
        return dist;
    }
};

} // namespace cbir
