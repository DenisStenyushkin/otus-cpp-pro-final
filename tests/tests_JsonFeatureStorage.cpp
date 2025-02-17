#include <gtest/gtest.h>

#include "FeatureVector.h"
#include "HsvHistogramFeatureVectorComputer.hpp"
#include "JsonFeatureStorage.hpp"
#include "ManhattanFeatureDistanceComputer.hpp"

TEST(JsonFeatureStorage, add_image_WorksOK) {
    cbir::HsvHistogramFeatureVectorComputer feature_computer{};
    cbir::ManhattanFeatureDistanceComputer<float, cbir::HSV_Features_D> distance_computer{};
    cbir::JsonFeatureStorage<float, cbir::HSV_Features_D> storage{feature_computer, distance_computer};

    cv::Mat image = cv::Mat::zeros(10, 10, CV_8UC3);
    storage.add_image("new_image", image);
    auto res = storage.get_features("new_image");

    ASSERT_FLOAT_EQ(cv::sum(res)[0], 5.0);
}
