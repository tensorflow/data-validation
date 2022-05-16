/* Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_data_validation/anomalies/metrics.h"

#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow_data_validation/anomalies/statistics_view_test_util.h"
#include "tensorflow_data_validation/anomalies/test_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {

namespace {

using tensorflow::metadata::v0::FeatureNameStatistics;
using testing::DatasetForTesting;
using testing::ParseTextProtoOrDie;

tensorflow::metadata::v0::FeatureNameStatistics
GetFeatureNameStatisticsWithTokens(const std::map<string, double>& tokens) {
  tensorflow::metadata::v0::FeatureNameStatistics result =
      ParseTextProtoOrDie<tensorflow::metadata::v0::FeatureNameStatistics>(
          R"(name: 'bar'
             type: STRING
             string_stats: {
               common_stats: {
                 num_missing: 0
                 max_num_values: 1
               }})");
  tensorflow::metadata::v0::StringStatistics* string_stats =
      result.mutable_string_stats();
  string_stats->set_unique(tokens.size());
  tensorflow::metadata::v0::RankHistogram* histogram =
      string_stats->mutable_rank_histogram();
  for (const auto& pair : tokens) {
    const string& feature_value = pair.first;
    const double feature_occurrences = pair.second;
    tensorflow::metadata::v0::RankHistogram::Bucket* bucket =
        histogram->add_buckets();
    *bucket->mutable_label() = feature_value;
    bucket->set_sample_count(feature_occurrences);
  }
  return result;
}

struct LInftyDistanceExample {
  string name;
  std::map<string, double> training;
  std::map<string, double> serving;
  double expected;
};

std::vector<LInftyDistanceExample> GetLInftyDistanceTests() {
  return {{"Two empty maps", {}, {}, 0.0},
          {"Normal distribution.",
           {{"hello", 0.1}, {"world", 0.9}},
           {{"hello", 0.3}, {"world", 0.7}},
           0.2},
          {"Missing value in both.",
           {{"b", 0.9}, {"c", 0.1}},
           {{"a", 0.3}, {"b", 0.7}},
           0.3},
          {"Missing value in both, flipped.",
           {{"a", 0.3}, {"b", 0.7}},
           {{"b", 0.9}, {"c", 0.1}},
           0.3}};
}

TEST(LInftyDistanceTest, All) {
  for (const auto& test : GetLInftyDistanceTests()) {
    const DatasetForTesting training(
        GetFeatureNameStatisticsWithTokens(test.training));
    const DatasetForTesting serving(
        GetFeatureNameStatisticsWithTokens(test.serving));
    const double result = LInftyDistance(training.feature_stats_view(),
                                         serving.feature_stats_view())
                              .second;
    EXPECT_NEAR(result, test.expected, 1e-5) << test.name;
  }
}

TEST(JensenShannonDivergence, SameStatistics) {
  const DatasetForTesting dataset(ParseTextProtoOrDie<FeatureNameStatistics>(R"(
    name: 'float'
    type: FLOAT
    num_stats {
      common_stats {}
      histograms {
        num_nan: 1
        buckets { low_value: 1.0 high_value: 2.3333333 sample_count: 2.9866667 }
        buckets {
          low_value: 2.3333333
          high_value: 3.6666667
          sample_count: 1.0066667
        }
        buckets { low_value: 3.6666667 high_value: 5.0 sample_count: 2.0066667 }
        type: STANDARD
      }
    })"));
  double result;
  TF_ASSERT_OK(JensenShannonDivergence(dataset.feature_stats_view(),
                                       dataset.feature_stats_view(), result));
  EXPECT_NEAR(result, 0.0, 1e-5);
}

TEST(JensenShannonDivergence, DifferentBucketBoundaries) {
  const DatasetForTesting dataset_1(
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'float'
        type: FLOAT
        num_stats {
          common_stats {}
          histograms {
            buckets { low_value: 1.0 high_value: 2.0 sample_count: 2.0 }
            buckets { low_value: 2.0 high_value: 3.0 sample_count: 2.0 }
            type: STANDARD
          }
        })"));
  const DatasetForTesting dataset_2(
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'float'
        type: FLOAT
        num_stats {
          common_stats {}
          histograms {
            buckets { low_value: 2.0 high_value: 4.0 sample_count: 2.0 }
            buckets { low_value: 4.0 high_value: 6.0 sample_count: 2.0 }
            type: STANDARD
          }
        })"));
  double result;
  TF_ASSERT_OK(JensenShannonDivergence(dataset_1.feature_stats_view(),
                                       dataset_2.feature_stats_view(), result));
  // Rebucketed and normalized histogram for dataset_1 is:
  // buckets { low_value: 1.0 high_value: 2.0 sample_count: 0.5 }
  // buckets { low_value: 2.0 high_value: 3.0 sample_count: 0.5 }
  // buckets { low_value: 3.0 high_value: 4.0 sample_count: 0 }
  // buckets { low_value: 4.0 high_value: 6.0 sample_count: 0 }
  // Rebucketed and normalized histogram for dataset_2 is:
  // buckets { low_value: 1.0 high_value: 2.0 sample_count: 0 }
  // buckets { low_value: 2.0 high_value: 3.0 sample_count: 0.25 }
  // buckets { low_value: 3.0 high_value: 4.0 sample_count: 0.25 }
  // buckets { low_value: 4.0 high_value: 6.0 sample_count: 0.5 }
  // Average distribution is:
  // buckets { low_value: 1.0 high_value: 2.0 sample_count: 0.25 }
  // buckets { low_value: 2.0 high_value: 3.0 sample_count: 0.325 }
  // buckets { low_value: 3.0 high_value: 4.0 sample_count: 0.125 }
  // buckets { low_value: 4.0 high_value: 6.0 sample_count: 0.25 }
  // JSD = (0.5*log(0.5/0.25) + 0.5*log(0.5/0.375))/2 + (0.25*log(0.25/0.375) +
  // 0.25*log(0.25/0.125) + 0.5*log(0.5/0.25))/2 = 0.65563906222
  EXPECT_NEAR(result, 0.65563906222, 1e-5);
}

TEST(JensenShannonDivergence, NoOverlap) {
  const DatasetForTesting dataset_1(
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'float'
        type: FLOAT
        num_stats {
          common_stats {}
          histograms {
            buckets { low_value: 0.0 high_value: 1.0 sample_count: 2.0 }
            buckets { low_value: 1.0 high_value: 2.0 sample_count: 2.0 }
            type: STANDARD
          }
        })"));
  const DatasetForTesting dataset_2(
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'float'
        type: FLOAT
        num_stats {
          common_stats {}
          histograms {
            buckets { low_value: 3.0 high_value: 4.0 sample_count: 2.0 }
            buckets { low_value: 4.0 high_value: 6.0 sample_count: 2.0 }
            type: STANDARD
          }
        })"));
  double result;
  TF_ASSERT_OK(JensenShannonDivergence(dataset_1.feature_stats_view(),
                                       dataset_2.feature_stats_view(), result));
  EXPECT_NEAR(result, 1, 1e-5);
}

TEST(JensenShannonDivergence, OneHasAllValuesInOneBucket) {
  const DatasetForTesting dataset_1(
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'float'
        type: FLOAT
        num_stats {
          common_stats {}
          histograms {
            buckets { low_value: 1.0 high_value: 1.0 sample_count: 4.0 }
            type: STANDARD
          }
        })"));
  const DatasetForTesting dataset_2(
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'float'
        type: FLOAT
        num_stats {
          common_stats {}
          histograms {
            buckets { low_value: 2.0 high_value: 4.0 sample_count: 2.0 }
            buckets { low_value: 4.0 high_value: 6.0 sample_count: 2.0 }
            type: STANDARD
          }
        })"));
  double result;
  TF_ASSERT_OK(JensenShannonDivergence(dataset_1.feature_stats_view(),
                                       dataset_2.feature_stats_view(), result));
  EXPECT_NEAR(result, 1, 1e-5);
}

TEST(JensenShannonDivergence, BothHaveAllValuesInOneBucket) {
  const DatasetForTesting dataset_1(
      ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
        name: 'float'
        type: FLOAT
        num_stats {
          common_stats {}
          histograms {
            buckets { low_value: 1.0 high_value: 1.0 sample_count: 4.0 }
            type: STANDARD
          }
        })pb"));
  const DatasetForTesting dataset_2(
      ParseTextProtoOrDie<FeatureNameStatistics>(R"pb(
        name: 'float'
        type: FLOAT
        num_stats {
          common_stats {}
          histograms {
            buckets { low_value: 2.0 high_value: 2.0 sample_count: 4.0 }
            type: STANDARD
          }
        })pb"));
  double result;
  TF_ASSERT_OK(JensenShannonDivergence(dataset_1.feature_stats_view(),
                                       dataset_2.feature_stats_view(), result));
  EXPECT_NEAR(result, 1, 1e-5);
}

TEST(JensenShannonDivergence, OneHasOneBucketTheOtherHasMany) {
  const DatasetForTesting dataset_1(
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'float'
        type: FLOAT
        num_stats {
          common_stats {}
          histograms {
            buckets { low_value: 10 high_value: 10 sample_count: 21 }
            type: STANDARD
          }
        })"));
  const DatasetForTesting dataset_2(
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'float'
        type: FLOAT
        num_stats {
          common_stats {}
          histograms {
            buckets { low_value: 0 high_value: 4 sample_count: 150 }
            buckets { low_value: 4 high_value: 12 sample_count: 200 }
            buckets { low_value: 12 high_value: 20 sample_count: 20 }
            type: STANDARD
          }
        })"));
  double result;
  TF_ASSERT_OK(JensenShannonDivergence(dataset_1.feature_stats_view(),
                                       dataset_2.feature_stats_view(), result));
  EXPECT_NEAR(result, 1, 1e-5);
}

TEST(JensenShannonDivergence, EmptyHistogram) {
  const DatasetForTesting empty_dataset_1(
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'float'
        type: FLOAT
        num_stats {
          common_stats {}
          histograms {
          }
        })"));
  const DatasetForTesting empty_dataset_2(
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'float'
        type: FLOAT
        num_stats {
          common_stats {}
          histograms {
            buckets { low_value: 0 high_value: 4 sample_count: 0 }
            type: STANDARD
          }
        })"));
  const DatasetForTesting non_empty_dataset(
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'float'
        type: FLOAT
        num_stats {
          common_stats {}
          histograms {
            buckets { low_value: 0 high_value: 4 sample_count: 10 }
            type: STANDARD
          }
        })"));
  double result;
  auto error =
      JensenShannonDivergence(empty_dataset_1.feature_stats_view(),
                              non_empty_dataset.feature_stats_view(), result);
  EXPECT_TRUE(tensorflow::errors::IsInvalidArgument(error));

  error = JensenShannonDivergence(non_empty_dataset.feature_stats_view(),
                                  empty_dataset_2.feature_stats_view(), result);
  EXPECT_TRUE(tensorflow::errors::IsInvalidArgument(error));
}

TEST(JensenShannonDivergence, WithNaNs) {
  const DatasetForTesting dataset_1(
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'float'
        type: FLOAT
        num_stats {
          common_stats {}
          histograms {
            num_nan: 1
            buckets { low_value: 1.0 high_value: 2.0 sample_count: 3.0 }
            type: STANDARD
          }
        })"));
  const DatasetForTesting dataset_2(
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'float'
        type: FLOAT
        num_stats {
          common_stats {}
          histograms {
            buckets { low_value: 1.0 high_value: 2.0 sample_count: 4.0 }
            type: STANDARD
          }
        })"));
  double result;
  TF_ASSERT_OK(JensenShannonDivergence(dataset_1.feature_stats_view(),
                                       dataset_2.feature_stats_view(), result));
  // Rebucketed and normalized histogram for dataset_1 is:
  // buckets { low_value: NaN high_value: Nan sample_count: 0.25 }
  // buckets { low_value: 1.0 high_value: 2.0 sample_count: 0.75 }
  // Rebucketed and normalized histogram for dataset_2 is:
  // buckets { low_value: NaN high_value: NaN sample_count: 0 }
  // buckets { low_value: 1.0 high_value: 2.0 sample_count: 1.0 }
  // Average distribution is:
  // buckets { low_value: NaN high_value: NaN sample_count: 0.125 }
  // buckets { low_value: 1.0 high_value: 2.0 sample_count: 0.875 }
  // JSD = (0.25*log(0.25/0.125) + 0.75*log(0.75/0.875))/2 +
  // (1.0*log(1.0/0.875))/2 = 0.13792538096
  EXPECT_NEAR(result, 0.13792538096, 1e-5);
}

TEST(JensenShannonDivergence, QuantileTypeHistograms) {
  const DatasetForTesting dataset(ParseTextProtoOrDie<
                                  FeatureNameStatistics>(R"pb(
    name: 'float'
    type: FLOAT
    num_stats {
      common_stats {}
      histograms {
        buckets { low_value: 1.0 high_value: 2.3333333 sample_count: 2.9866667 }
        buckets {
          low_value: 2.3333333
          high_value: 3.6666667
          sample_count: 1.0066667
        }
        buckets { low_value: 3.6666667 high_value: 5.0 sample_count: 2.0066667 }
        type: QUANTILES
      }
    })pb"));
  double result;
  auto error = JensenShannonDivergence(dataset.feature_stats_view(),
                                       dataset.feature_stats_view(), result);
  EXPECT_TRUE(tensorflow::errors::IsInvalidArgument(error));
}

}  // namespace

}  // namespace data_validation

}  // namespace tensorflow
