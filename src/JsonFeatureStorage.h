#pragma once


#include <string>
#include <vector>

#include "opencv2/core.hpp"

#include "IFeatureStorage.h"
#include "IFeatureVectorComputer.h"
#include "IFeatureDistanceComputer.h"
#include "FeatureVector.h"


namespace cbir {

template<typename dtype, size_t D>
class JsonFeatureStorage : IFeatureStorage<dtype, D>
{
public:
    JsonFeatureStorage(const IFeatureVectorComputer& feature_vector_computer,
                       const IFeatureDistanceComputer& feature_distance_computer);
    ~JsonFeatureStorage();

    void add_image(const std::string& key, const cv::Mat& image) override;
    FeatureVector<dtype, D> get_features(const std::string& key) override;
    double compute_feature_distance(const std::string& key1, const std::string& key2) override;
    std::vector<std::pair<std::string, double>> find_nearest(const std::string& key, size_t n) override;
    void save() override;
};

} // namespace cbir
