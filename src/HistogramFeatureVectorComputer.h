#pragma once

#include "opencv2/core.hpp"

#include "IFeatureVectorComputer.h"
#include "FeatureVector.h"

namespace cbir
{

template<typename dtype, size_t D>
class HistogramFeatureVectorComputer : IFeatureVectorComputer<dtype> {
public:
    HistogramFeatureVectorComputer();

    FeatureVector<dtype, D> compute(const cv2::Mat& image) override;
};

} // namespace cbir
