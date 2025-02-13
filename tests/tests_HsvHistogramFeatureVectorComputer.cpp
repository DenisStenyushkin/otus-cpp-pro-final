#include <gtest/gtest.h>
#include "opencv2/core.hpp"

#include "HsvHistogramFeatureVectorComputer.hpp"


TEST(HsvHistogramFeatureVectorComputer, WorksOK) {
    cv::Mat zeros = cv::Mat::zeros(10, 10, CV_8UC3);
    cbir::HsvHistogramFeatureVectorComputer computer;
    
    auto res = computer.compute(zeros);

    ASSERT_FLOAT_EQ(cv::sum(res)[0], 5.0);
}
