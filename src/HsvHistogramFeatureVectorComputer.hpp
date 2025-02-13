#pragma once

#include <vector>
#include <tuple>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include "IFeatureVectorComputer.h"
#include "FeatureVector.h"

namespace cbir
{

constexpr size_t HSV_Features_D = 288 * 5;

class HsvHistogramFeatureVectorComputer : IFeatureVectorComputer<float, HSV_Features_D> {
public:

    FeatureVector<float, HSV_Features_D> compute(const cv::Mat& image) override {
        cv::Mat image_hsv;
        cv::cvtColor(image, image_hsv, cv::COLOR_BGR2HSV);

        std::vector<cv::Mat> segment_features;
        segment_features.reserve(5);

        auto h = image.rows;
        auto w = image.cols;

        int cx = w / 2;
        int cy = h / 2;
        std::vector<std::tuple<int, int, int, int>> segments{
            std::make_tuple(0, cx, 0, cy),
            std::make_tuple(cx, w, 0, cy),
            std::make_tuple(cx, w, cy, h),
            std::make_tuple(0, cx, cy, h)
        };

        int ax_x = w * 3 / 8; // (0.75 * w) / 2
        int ax_y = h * 3 / 8;
        cv::Mat ellipseMask = cv::Mat::zeros(h, w, CV_8U);
        cv::ellipse(ellipseMask, {cx, cy}, {ax_x, ax_y}, 0, 0, 360, 255, -1);

        for (const auto [start_x, end_x, start_y, end_y] : segments) {
            cv::Mat corner_mask = cv::Mat::zeros(image.rows, image.cols, CV_8U);
            cv::rectangle(corner_mask, {start_x, start_y}, {end_x, end_y}, 255, -1);
            cv::subtract(corner_mask, ellipseMask, corner_mask);

            cv::Mat hist = claculate_hist(image, corner_mask);
            segment_features.push_back(hist);
        }

        cv::Mat hist = claculate_hist(image, ellipseMask);
        segment_features.push_back(hist);

        FeatureVector<float, HSV_Features_D> features;
        size_t idx = 0;
        for (const auto& hist: segment_features) {
            for (size_t j = 0; j < hist.size().width; ++j) {
                features[idx] = hist.at<float>(j);
                ++idx;
            }
        }

        return features;
    }

private:
    cv::Mat claculate_hist(const cv::Mat& image, const cv::Mat& mask) {
        cv::Mat hist;
        int channels[] = {0, 1, 2};
        int bins[] = {8, 12, 3};
        float h_ranges[] = { 0, 180 };
        float s_ranges[] = { 0, 256 };
        float v_ranges[] = { 0, 256 };
        const float* ranges[] = { h_ranges, s_ranges, v_ranges };
        cv::calcHist(&image, 1, channels, mask, hist, 3, bins, ranges);

        [[maybe_unused]] auto t1 = hist.type();

        cv::Mat hist_norm;
        cv::normalize(hist, hist_norm);

        cv::Mat hist_norm_flat = hist_norm.reshape(1, 1);

        [[maybe_unused]] auto t2 = hist_norm_flat.type();

        return hist_norm_flat;
    }
};

} // namespace cbir
