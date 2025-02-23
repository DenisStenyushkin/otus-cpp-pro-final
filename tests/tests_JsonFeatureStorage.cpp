#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "FeatureVector.h"
#include "HsvHistogramFeatureVectorComputer.hpp"
#include "JsonFeatureStorage.hpp"
#include "ManhattanFeatureDistanceComputer.hpp"

TEST(JsonFeatureStorage, add_image_WorksOK) {
    cbir::HsvHistogramFeatureVectorComputer feature_computer{};
    cbir::ManhattanFeatureDistanceComputer<float, cbir::HSV_Features_D> distance_computer{};
    cbir::JsonFeatureStorage<float, cbir::HSV_Features_D> storage{"dummy.json", feature_computer, distance_computer};

    cv::Mat image = cv::Mat::zeros(10, 10, CV_8UC3);
    storage.add_image("new_image", image);
    auto res = storage.get_features("new_image");

    ASSERT_FLOAT_EQ(cv::sum(res)[0], 5.0);
}

TEST(JsonFeatureStorage, compute_feature_distance_WorksOK) {
    cbir::HsvHistogramFeatureVectorComputer feature_computer{};
    cbir::ManhattanFeatureDistanceComputer<float, cbir::HSV_Features_D> distance_computer{};
    cbir::JsonFeatureStorage<float, cbir::HSV_Features_D> storage{"dummy.json", feature_computer, distance_computer};

    cv::Mat image1 = cv::Mat::zeros(10, 10, CV_8UC3);
    storage.add_image("image1", image1);

    cv::Mat image2 = cv::Mat::ones(10, 10, CV_8UC3) * 255;
    storage.add_image("image2", image2);

    double distance = storage.compute_feature_distance("image1", "image2");
    ASSERT_FLOAT_EQ(distance, 5.0);
}

TEST(JsonFeatureStorage, find_nearest_WorksOK) {
    cbir::HsvHistogramFeatureVectorComputer feature_computer{};
    cbir::ManhattanFeatureDistanceComputer<float, cbir::HSV_Features_D> distance_computer{};
    cbir::JsonFeatureStorage<float, cbir::HSV_Features_D> storage{"dummy.json", feature_computer, distance_computer};

    std::vector<std::string> img_names = { "100301.jpg", "100302.jpg", "100500.jpg", "100600.jpg", 
                                           "101700.jpg", "102301.jpg", "104801.jpg",  };
    std::vector<cv::Mat> images{};
    for (const std::string& fname: img_names) {
        images.push_back(cv::imread("/workspaces/otus-cpp-pro-final/tests/data/" + fname));
    }

    for (size_t i = 0; i < img_names.size(); ++i) {
        storage.add_image(img_names[i], images[i]);
    }

    auto res = storage.find_nearest(img_names[0], 2);
    
    ASSERT_TRUE(res[0].first == img_names[1]);
    ASSERT_TRUE(res[1].first == img_names[2]);
}

TEST(JsonFeatureStorage, find_nearest_nokey_WorksOK) {
    cbir::HsvHistogramFeatureVectorComputer feature_computer{};
    cbir::ManhattanFeatureDistanceComputer<float, cbir::HSV_Features_D> distance_computer{};
    cbir::JsonFeatureStorage<float, cbir::HSV_Features_D> storage{"dummy.json", feature_computer, distance_computer};

    std::vector<std::string> img_names = { "100301.jpg", "100302.jpg", "100500.jpg", "100600.jpg", 
                                           "101700.jpg", "102301.jpg", "104801.jpg",  };
    std::vector<cv::Mat> images{};
    for (const std::string& fname: img_names) {
        images.push_back(cv::imread("/workspaces/otus-cpp-pro-final/tests/data/" + fname));
    }

    for (size_t i = 0; i < img_names.size(); ++i) {
        storage.add_image(img_names[i], images[i]);
    }

    EXPECT_THROW(storage.find_nearest("not_exists", 2), std::runtime_error);
}

TEST(JsonFeatureStorage, save_WorksOK) {
    cbir::HsvHistogramFeatureVectorComputer feature_computer{};
    cbir::ManhattanFeatureDistanceComputer<float, cbir::HSV_Features_D> distance_computer{};
    cbir::JsonFeatureStorage<float, cbir::HSV_Features_D> storage{"test_storage.json", feature_computer, distance_computer};

    std::vector<std::string> img_names = { "100301.jpg", "100302.jpg", "100500.jpg", "100600.jpg", 
                                           "101700.jpg", "102301.jpg", "104801.jpg",  };
    std::vector<cv::Mat> images{};
    for (const std::string& fname: img_names) {
        images.push_back(cv::imread("/workspaces/otus-cpp-pro-final/tests/data/" + fname));
    }

    for (size_t i = 0; i < img_names.size(); ++i) {
        storage.add_image(img_names[i], images[i]);
    }

    storage.save();
}

TEST(JsonFeatureStorage, ctor_WorksOK) {
    cbir::HsvHistogramFeatureVectorComputer feature_computer{};
    cbir::ManhattanFeatureDistanceComputer<float, cbir::HSV_Features_D> distance_computer{};
    cbir::JsonFeatureStorage<float, cbir::HSV_Features_D> storage{"test_storage.json", feature_computer, distance_computer};
}
