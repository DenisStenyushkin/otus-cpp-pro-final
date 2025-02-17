#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

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
    JsonFeatureStorage(const IFeatureVectorComputer<dtype, D>& feature_vector_computer,
                       const IFeatureDistanceComputer<dtype, D>& feature_distance_computer)
        : feature_computer_{feature_vector_computer}, distance_computer_{feature_distance_computer}
    {
        read_from_file();
    }

    ~JsonFeatureStorage() override {
        save();
    }

    void add_image(const std::string& key, const cv::Mat& image) override {
        auto features = feature_computer_.compute(image);
        features_[key] = features; // TODO: move
    }

    const FeatureVector<dtype, D>& get_features(const std::string& key) override {
        check_key(key);

        return features_[key];
    }

    double compute_feature_distance(const std::string& key1, const std::string& key2) override {
        check_key(key1);
        check_key(key2);

        return distance_computer_.compute(features_[key1], features_[key2]);
    }

    std::vector<std::pair<std::string, double>> find_nearest(const std::string& key, size_t n) override {
        std::vector<std::pair<std::string, double>> distances;
        std::transform(features_.begin(), features_.end(), std::back_inserter(distances),
                        [this, &key](auto& pair) {
                            std::pair<std::string, double> result;
                            result.first = pair.first;
                            result.second = distance_computer_.compute(features_[key], features_[pair.first]);
                            return result;
                        });

        std::partial_sort(distances.begin(), distances.begin() + n, distances.end(),
                          [](const auto& e1, const auto& e2) { return e1.second < e2.second; });

        std::vector<std::pair<std::string, double>> result{distances.begin(), distances.begin() + n};

        return result;
    }

    void save() override {
    }

private:
    const IFeatureVectorComputer<dtype, D>& feature_computer_;
    const IFeatureDistanceComputer<dtype, D>& distance_computer_;
    const std::string filename_ = "storage.json";
    std::unordered_map<std::string, FeatureVector<dtype, D>> features_;

    void read_from_file() {
    }

    void check_key(const std::string& key) {
        if (features_.find(key) == features_.end()) {
            throw std::runtime_error("No features for key " + key);
        }
    }
};

} // namespace cbir
