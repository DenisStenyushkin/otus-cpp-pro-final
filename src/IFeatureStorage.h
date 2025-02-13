#pragma once


#include <string>
#include <vector>

#include "opencv2/core.hpp"

#include "FeatureVector.h"


namespace cbir {

template<typename dtype, size_t D>
class IFeatureStorage
{
public:
    virtual void add_image(const std::string& key, const cv::Mat& image) = 0;
    virtual FeatureVector<dtype, D> get_features(const std::string& key) = 0;
    virtual double compute_feature_distance(const std::string& key1, const std::string& key2) = 0;
    virtual std::vector<std::pair<std::string, double>> find_nearest(const std::string& key, size_t n) = 0;
    virtual void save() = 0;
};

} // namespace cbir
