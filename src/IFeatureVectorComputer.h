#pragma once

#include "opencv2/core.hpp"

#include "FeatureVector.h"

namespace cbir {

template<typename dtype, size_t D>
class IFeatureVectorComputer {
public:
    virtual FeatureVector<dtype, D> compute(const cv::Mat& image) const = 0;
};

} // namespace cbir
