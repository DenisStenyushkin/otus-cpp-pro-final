#include <gtest/gtest.h>

#include "FeatureVector.h"
#include "ManhattanFeatureDistanceComputer.hpp"

TEST(ManhattanFeatureDistanceComputer, WorksOK) {
    cbir::FeatureVector<double, 5> v1{1.0, 2.5, 3.2, -10.1, 0.0};
    cbir::FeatureVector<double, 5> v2{1.0, 2.0, 3.5, 0.0, 2.0};

    auto dist = cbir::ManhattanFeatureDistanceComputer<double, 5>().compute(v1, v2);
    ASSERT_NEAR(dist, 12.9, 1e-5);
}
