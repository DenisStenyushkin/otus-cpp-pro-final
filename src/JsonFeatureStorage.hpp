#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

#include "json.hpp"
#include "opencv2/core.hpp"

#include "IFeatureStorage.h"
#include "IFeatureVectorComputer.h"
#include "IFeatureDistanceComputer.h"
#include "FeatureVector.h"


using json = nlohmann::json;

namespace cbir {

template<typename dtype, size_t D>
class JsonFeatureStorage : IFeatureStorage<dtype, D>
{
public:
    JsonFeatureStorage(std::string filename,
                       const IFeatureVectorComputer<dtype, D>& feature_vector_computer,
                       const IFeatureDistanceComputer<dtype, D>& feature_distance_computer)
        : filename_{filename}, feature_computer_{feature_vector_computer}, distance_computer_{feature_distance_computer}
    {
        if (std::filesystem::exists(std::filesystem::path{filename_})) {
            read_from_file();
        }
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
        check_key(key);

        std::vector<std::pair<std::string, double>> distances;
        std::transform(features_.begin(), features_.end(), std::back_inserter(distances),
                        [this, &key](auto& pair) {
                            std::pair<std::string, double> result;
                            result.first = pair.first;
                            result.second = distance_computer_.compute(features_[key], features_[pair.first]);
                            return result;
                        });

        // distances[0] will be image itself, so we ignore account for it here and ignore when copying results
        std::partial_sort(distances.begin(), distances.begin() + n + 1, distances.end(),
                          [](const auto& e1, const auto& e2) { return e1.second < e2.second; });
        std::vector<std::pair<std::string, double>> result{distances.begin() + 1, distances.begin() + n + 1};

        return result;
    }

    void save() override {
        json j(features_);
        std::ofstream file{filename_};
        file << std::setw(4) << j;
    }

private:
    const std::string filename_;
    const IFeatureVectorComputer<dtype, D>& feature_computer_;
    const IFeatureDistanceComputer<dtype, D>& distance_computer_;
    std::unordered_map<std::string, FeatureVector<dtype, D>> features_;

    void read_from_file() {
        json j;
        std::ifstream file{filename_};

        file >> j;
        features_ = j.get<std::unordered_map<std::string, FeatureVector<dtype, D>>>();
    }

    void check_key(const std::string& key) {
        if (features_.find(key) == features_.end()) {
            throw std::runtime_error("No features for key " + key);
        }
    }
};

} // namespace cbir
