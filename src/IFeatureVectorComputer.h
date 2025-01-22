#pragma once

#include "opencv2/core.hpp"

#include "FeatureVector.h"

namespace cbir {

template<typename dtype, size_t D>
class IFeatureVectorComputer {
    virtual FeatureVector<dtype, D> compute(const cv2::Mat& image) = 0;
};

} // namespace cbir
