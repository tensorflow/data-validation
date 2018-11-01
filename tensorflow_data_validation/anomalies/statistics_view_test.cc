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

#include "tensorflow_data_validation/anomalies/statistics_view.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/optional.h"
#include "tensorflow_data_validation/anomalies/statistics_view_test_util.h"
#include "tensorflow_data_validation/anomalies/test_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {

using ::tensorflow::metadata::v0::DatasetFeatureStatistics;
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using testing::ParseTextProtoOrDie;

TEST(DatasetStatsView, Environment) {
  const DatasetFeatureStatistics current =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features {
          name: 'bar'
          type: FLOAT
          num_stats: {
            common_stats: { num_missing: 3 min_num_values: 3 max_num_values: 7 }
          }
        })");

  DatasetStatsView view(current, false, "environment_name",
                        std::shared_ptr<DatasetStatsView>(),
                        std::shared_ptr<DatasetStatsView>());

  EXPECT_TRUE(view.environment());
  EXPECT_EQ("environment_name", *view.environment());

  DatasetStatsView view_no_environment(current, false);
  EXPECT_FALSE(view_no_environment.environment());
}

TEST(FeatureStatsView, Environment) {
  const DatasetFeatureStatistics current =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features {
          name: 'bar'
          type: FLOAT
          num_stats: {
            common_stats: { num_missing: 3 min_num_values: 3 max_num_values: 7 }
          }
        })");

  DatasetStatsView view(current, false, "environment_name",
                        std::shared_ptr<DatasetStatsView>(),
                        std::shared_ptr<DatasetStatsView>());

  EXPECT_TRUE(view.GetByPath(Path({"bar"}))->environment());
  EXPECT_EQ("environment_name", *view.GetByPath(Path({"bar"}))->environment());
}

TEST(FeatureStatsView, Previous) {
  const DatasetFeatureStatistics current =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features {
          name: 'bar'
          type: FLOAT
          num_stats: {
            common_stats: { num_missing: 3 min_num_values: 3 max_num_values: 7 }
          }
        })");
  const DatasetFeatureStatistics previous = ParseTextProtoOrDie<
      DatasetFeatureStatistics>(R"(
    features {
      name: 'bar'
      type: FLOAT
      num_stats: {
        common_stats: { num_missing: 3 min_num_values: 3 max_num_values: 10 }
      }
    })");

  std::shared_ptr<DatasetStatsView> previous_view =
      std::make_shared<DatasetStatsView>(previous);

  DatasetStatsView view(current, false, "environment_name", previous_view,
                        std::shared_ptr<DatasetStatsView>());

  EXPECT_EQ(10, view.GetByPath(Path({"bar"}))->GetPrevious()->max_num_values());
  EXPECT_EQ(10, view.GetPrevious()->GetByPath(Path({"bar"}))->max_num_values());
  DatasetStatsView view_no_previous(current, false);
  EXPECT_FALSE(view_no_previous.GetServing());
  EXPECT_FALSE(view_no_previous.GetByPath(Path({"bar"}))->GetServing());
}

TEST(FeatureStatsView, Serving) {
  const DatasetFeatureStatistics current =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features {
          name: 'bar'
          type: FLOAT
          num_stats: {
            common_stats: { num_missing: 3 min_num_values: 3 max_num_values: 7 }
          }
        })");
  const DatasetFeatureStatistics serving = ParseTextProtoOrDie<
      DatasetFeatureStatistics>(R"(
    features {
      name: 'bar'
      type: FLOAT
      num_stats: {
        common_stats: { num_missing: 3 min_num_values: 3 max_num_values: 10 }
      }
    })");

  std::shared_ptr<DatasetStatsView> serving_view =
      std::make_shared<DatasetStatsView>(serving);

  DatasetStatsView view(current, false, "environment_name",
                        std::shared_ptr<DatasetStatsView>(), serving_view);

  EXPECT_EQ(10, view.GetServing()->GetByPath(Path({"bar"}))->max_num_values());
  EXPECT_EQ(10, view.GetByPath(Path({"bar"}))->GetServing()->max_num_values());
  DatasetStatsView view_no_serving(current, false);
  EXPECT_FALSE(view_no_serving.GetServing());
  EXPECT_FALSE(view_no_serving.GetByPath(Path({"bar"}))->GetServing());
}

TEST(FeatureStatsView, Name) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: FLOAT
        num_stats: {
          common_stats: { num_missing: 3 min_num_values: 3 max_num_values: 7 }
        })");

  EXPECT_EQ("bar",
            testing::DatasetForTesting(input).feature_stats_view().name());
}

TEST(FeatureStatsView, Type) {
  const testing::DatasetForTesting dataset(
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'optional_float'
        type: FLOAT
        num_stats: {
          common_stats: {
            num_missing: 0
            num_non_missing: 10
            min_num_values: 1
            max_num_values: 1
          }
        })"));
  EXPECT_EQ(tensorflow::metadata::v0::FeatureNameStatistics::FLOAT,
            dataset.feature_stats_view().type());
}

TEST(FeatureStatsView, GetNumPresent) {
  const testing::DatasetForTesting dataset(
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'optional_float'
        type: FLOAT
        num_stats: {
          common_stats: {
            num_missing: 0
            num_non_missing: 10
            min_num_values: 1
            max_num_values: 1
          }
        })"));
  EXPECT_EQ(10.0, dataset.feature_stats_view().GetNumPresent());
}

TEST(FeatureStatsView, GetNumPresentForStruct) {
  const testing::DatasetForTesting dataset(
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'the_struct'
        type: STRUCT
        struct_stats: {
          common_stats: {
            num_missing: 0
            num_non_missing: 10
            min_num_values: 1
            max_num_values: 1
          }
        })"));
  EXPECT_EQ(10.0, dataset.feature_stats_view().GetNumPresent());
}

TEST(FeatureStatsView, MinNumValues) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: FLOAT
        num_stats: {
          common_stats: { num_missing: 3 min_num_values: 3 max_num_values: 7 }
        })");

  EXPECT_THAT(
      3,
      testing::DatasetForTesting(input).feature_stats_view().min_num_values());
}

TEST(FeatureStatsView, MaxNumValues) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: FLOAT
        num_stats: {
          common_stats: { num_missing: 3 min_num_values: 3 max_num_values: 7 }
        })");

  EXPECT_THAT(
      7,
      testing::DatasetForTesting(input).feature_stats_view().max_num_values());
}

TEST(FeatureStatsView, GetNumExamples) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: FLOAT
        num_stats: {
          common_stats: {
            num_missing: 3
            num_non_missing: 6
            min_num_values: 3
            max_num_values: 7
          }
        })");
  EXPECT_EQ(
      9,
      testing::DatasetForTesting(input).feature_stats_view().GetNumExamples());
}

TEST(DatasetStatsView, GetParentGetPath) {
  const DatasetFeatureStatistics input =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: 'foo.bar'
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 6
              min_num_values: 3
              max_num_values: 7
            }
          }
        }
        features: {
          name: 'foo'
          type: STRUCT
          struct_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 6
              min_num_values: 1
              max_num_values: 1
            }
          }
        })");
  DatasetStatsView stats(input);
  absl::optional<FeatureStatsView> actual =
      stats.GetByPath(Path({"foo", "bar"}));
  ASSERT_TRUE(actual);
  absl::optional<FeatureStatsView> parent = actual->GetParent();
  ASSERT_TRUE(parent);
  EXPECT_EQ(parent->name(), "foo");
  EXPECT_EQ(Path({"foo", "bar"}), actual->GetPath());
}

// foo is not a parent of foo.bar, as they are both floats.
TEST(DatasetStatsView, GetParentFalsePositiveGetPath) {
  const DatasetFeatureStatistics input =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: 'foo.bar'
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 6
              min_num_values: 3
              max_num_values: 7
            }
          }
        }
        features: {
          name: 'foo'
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 6
              min_num_values: 3
              max_num_values: 7
            }
          }
        })");
  DatasetStatsView stats(input);
  absl::optional<FeatureStatsView> actual = stats.GetByPath(Path({"foo.bar"}));
  EXPECT_TRUE(actual);
  absl::optional<FeatureStatsView> parent = actual->GetParent();
  EXPECT_FALSE(parent);
  EXPECT_EQ(Path({"foo.bar"}), actual->GetPath());
}

TEST(DatasetStatsView, GetRootFeatures) {
  const DatasetFeatureStatistics input =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: 'foo.bar'
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 6
              min_num_values: 3
              max_num_values: 7
            }
          }
        }
        features: {
          name: 'foo'
          type: STRUCT
          struct_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 6
              min_num_values: 1
              max_num_values: 1
            }
          }
        })");
  DatasetStatsView stats(input);
  std::vector<FeatureStatsView> roots = stats.GetRootFeatures();
  ASSERT_EQ(roots.size(), 1);
  EXPECT_EQ(roots[0].name(), "foo");
}

TEST(FeatureStatsView, GetNumExamplesWeighted) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: FLOAT
        num_stats: {
          min: 8.0
          common_stats: {
            num_missing: 3
            num_non_missing: 7
            min_num_values: 3
            max_num_values: 7
            weighted_common_stats: { num_missing: 3.0 num_non_missing: 5.5 }
          }
        })");

  EXPECT_EQ(10.0, testing::DatasetForTesting(input, false)
                      .feature_stats_view()
                      .GetNumExamples());
  EXPECT_EQ(8.5, testing::DatasetForTesting(input, true)
                     .dataset_stats_view()
                     .GetNumExamples());
}

TEST(FeatureStatsView, GetStringValuesWithCountsBasicTest) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: { num_missing: 3 max_num_values: 2 }
          unique: 3
          rank_histogram: {
            buckets: { label: "foo" sample_count: 1.5 }
            buckets: { label: "bar" sample_count: 2 }
            buckets: { label: "baz" sample_count: 3 }
          }
        })");

  const std::map<string, double> actual = testing::DatasetForTesting(input)
                                              .feature_stats_view()
                                              .GetStringValuesWithCounts();
  EXPECT_THAT(actual,
              ::testing::UnorderedElementsAre(::testing::Pair("foo", 1.5),
                                              ::testing::Pair("bar", 2),
                                              ::testing::Pair("baz", 3)));
}

TEST(FeatureStatsView, GetStringValuesWithCountsEmptyResult) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        num_stats: { common_stats: { num_missing: 3 max_num_values: 2 } })");

  const std::map<string, double> actual = testing::DatasetForTesting(input)
                                              .feature_stats_view()
                                              .GetStringValuesWithCounts();
  EXPECT_THAT(actual, ::testing::IsEmpty());
}

TEST(FeatureStatsView, GetStringValuesWithCountsNotString) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        num_stats: { common_stats: { num_non_missing: 4 num_missing: 1 } }
        type: STRING)");

  const std::map<string, double> actual = testing::DatasetForTesting(input)
                                              .feature_stats_view()
                                              .GetStringValuesWithCounts();
  EXPECT_THAT(actual, ::testing::IsEmpty());
}

TEST(FeatureStatsView, HasInvalidUTF8Strings) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: { num_missing: 3 max_num_values: 2 }
          unique: 3
          rank_histogram: {
            buckets: { label: "__BYTES_VALUE__" sample_count: 1.5 }
            buckets: { label: "bar" sample_count: 2 }
            buckets: { label: "baz" sample_count: 3 }
          }
        })");

  EXPECT_TRUE(testing::DatasetForTesting(input)
                  .feature_stats_view()
                  .HasInvalidUTF8Strings());
}

TEST(FeatureStatsView, HasInvalidUTF8StringsFalse) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: { num_missing: 3 max_num_values: 2 }
          unique: 3
          rank_histogram: {
            buckets: { label: "foo" sample_count: 1.5 }
            buckets: { label: "bar" sample_count: 2 }
            buckets: { label: "baz" sample_count: 3 }
          }
        })");

  EXPECT_FALSE(testing::DatasetForTesting(input)
                   .feature_stats_view()
                   .HasInvalidUTF8Strings());
}

TEST(FeatureStatsView, GetStringValuesNoStringTest) {
  const tensorflow::metadata::v0::FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        num_stats: { common_stats: { num_non_missing: 4 num_missing: 1 } }
        type: STRING)");

  const std::vector<string> actual =
      testing::DatasetForTesting(input).feature_stats_view().GetStringValues();
  EXPECT_THAT(actual, ::testing::IsEmpty());
}

TEST(FeatureStatsView, GetStringValuesBasic) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: { num_missing: 3 max_num_values: 2 }
          unique: 3
          rank_histogram: {
            buckets: { label: "foo" }
            buckets: { label: "bar" }
            buckets: { label: "baz" }
          }
        })");
  const std::vector<string> actual =
      testing::DatasetForTesting(input).feature_stats_view().GetStringValues();
  EXPECT_THAT(actual, ::testing::ElementsAre("bar", "baz", "foo"));
}

// Try get_missing with the weight_by field.
TEST(FeatureStatsView, GetStringValuesWithCountsWeighted) {
  const tensorflow::metadata::v0::FeatureNameStatistics stats =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: { num_missing: 3 max_num_values: 2 }
          unique: 3
          weighted_string_stats: {
            rank_histogram: {
              buckets: { label: "alpha" sample_count: 123 }
              buckets: { label: "beta" sample_count: 234 }
            }
          }
        })");
  const testing::DatasetForTesting dataset(stats, true);
  const std::map<string, double> result =
      dataset.feature_stats_view().GetStringValuesWithCounts();
  EXPECT_THAT(result, ::testing::ElementsAre(::testing::Pair("alpha", 123),
                                             ::testing::Pair("beta", 234)));
}

// Return the numeric stats, or an empty object if no numeric stats exist.
TEST(FeatureStatsView, NumStats) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: FLOAT
        num_stats: {
          min: 8.0
          common_stats: { num_missing: 3 min_num_values: 3 max_num_values: 7 }
        })");

  EXPECT_EQ(
      8.0,
      testing::DatasetForTesting(input).feature_stats_view().num_stats().min());
  const FeatureNameStatistics missing =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        type: STRING
        string_stats: {
          common_stats: { num_missing: 3 max_num_values: 2 }
          unique: 3
          weighted_string_stats: {
            rank_histogram: {
              buckets: { label: "alpha" sample_count: 123 }
              buckets: { label: "beta" sample_count: 234 }
            }
          }
        })");
  EXPECT_EQ(0.0, testing::DatasetForTesting(missing)
                     .feature_stats_view()
                     .num_stats()
                     .min());
}

TEST(FeatureStatsView, GetNumMissing) {
  const DatasetFeatureStatistics input =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 10
        weighted_num_examples: 9.75
        features: {
          name: 'bar'
          type: FLOAT
          num_stats: {
            min: 8.0
            common_stats: {
              num_missing: 3
              num_non_missing: 7
              min_num_values: 3
              max_num_values: 7
              weighted_common_stats: { num_non_missing: 5.5 num_missing: 4.25 }
            }
          }
        })");

  EXPECT_EQ(3.0, DatasetStatsView(input).features()[0].GetNumMissing());
  EXPECT_EQ(4.25, testing::DatasetForTesting(input, true)
                      .feature_stats_view()
                      .GetNumMissing());
}

TEST(FeatureStatsView, GetFeatureType) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRUCT
        struct_stats: {
          common_stats: {
            num_missing: 0
            num_non_missing: 6
            min_num_values: 1
            max_num_values: 1
          }
        })");

  EXPECT_EQ(
      tensorflow::metadata::v0::STRUCT,
      testing::DatasetForTesting(input).feature_stats_view().GetFeatureType());
}

TEST(FeatureStatsView, GetFractionPresent) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: FLOAT
        num_stats: {
          min: 8.0
          common_stats: {
            num_missing: 2
            num_non_missing: 6
            min_num_values: 3
            max_num_values: 7
            weighted_common_stats: { num_non_missing: 2.5 num_missing: 7.5 }
          }
        })");

  absl::optional<double> result_a = testing::DatasetForTesting(input)
                                        .feature_stats_view()
                                        .GetFractionPresent();
  EXPECT_EQ(0.75, *result_a);
  absl::optional<double> result_b = testing::DatasetForTesting(input, true)
                                        .feature_stats_view()
                                        .GetFractionPresent();
  EXPECT_EQ(0.25, *result_b);
}

TEST(FeatureStatsView, GetTotalValueCountInExamples) {
  // double GetNumMissing() const;
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: FLOAT
        num_stats: {
          min: 8.0
          common_stats: {
            avg_num_values: 0.5
            num_missing: 2
            num_non_missing: 8
            min_num_values: 3
            max_num_values: 7
            tot_num_values: 4
            weighted_common_stats: { num_non_missing: 5.5 }
          }
        })");

  EXPECT_EQ(4.0, testing::DatasetForTesting(input)
                     .feature_stats_view()
                     .GetTotalValueCountInExamples());
}

// Older protos may not have tot_num_values set.
TEST(FeatureStatsView, GetTotalValueCountInExamplesOld) {
  // double GetNumMissing() const;
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: FLOAT
        num_stats: {
          min: 8.0
          common_stats: {
            avg_num_values: 0.5
            num_missing: 2
            num_non_missing: 8
            min_num_values: 3
            max_num_values: 7
            weighted_common_stats: { num_non_missing: 5.5 }
          }
        })");

  EXPECT_EQ(4.0, testing::DatasetForTesting(input)
                     .feature_stats_view()
                     .GetTotalValueCountInExamples());
}

TEST(DatasetStatsView, Features) {
  // double GetNumMissing() const;
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: FLOAT
        num_stats: {
          min: 8.0
          common_stats: {
            avg_num_values: 0.5
            num_missing: 2
            num_non_missing: 8
            min_num_values: 3
            max_num_values: 7
            weighted_common_stats: { num_non_missing: 5.5 }
          }
        })");

  EXPECT_EQ(
      1,
      testing::DatasetForTesting(input).dataset_stats_view().features().size());
}

TEST(DatasetStatsView, GetNumExamples) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: FLOAT
        num_stats: {
          common_stats: {
            num_missing: 3
            num_non_missing: 6
            min_num_values: 3
            max_num_values: 7
          }
        })");
  EXPECT_EQ(
      9,
      testing::DatasetForTesting(input).dataset_stats_view().GetNumExamples());
}

TEST(DatasetStatsView, GetNumExamplesWeighted) {
  // double GetNumMissing() const;
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: FLOAT
        num_stats: {
          min: 8.0
          common_stats: {
            num_missing: 3
            num_non_missing: 7
            min_num_values: 3
            max_num_values: 7
            weighted_common_stats: { num_non_missing: 5.5 num_missing: 3.0 }
          }
        })");
  //  weighted_num_examples = num_non_missing + num_missing = 5.5 + 3.0 = 8.5
  EXPECT_EQ(8.5, testing::DatasetForTesting(input, true)
                     .dataset_stats_view()
                     .GetNumExamples());
}

TEST(DatasetStatsView, ByWeight) {
  const tensorflow::metadata::v0::FeatureNameStatistics feature_stats =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'optional_float'
        type: FLOAT
        num_stats: {
          common_stats: {
            num_missing: 0
            num_non_missing: 10
            min_num_values: 1
            max_num_values: 1
          }
        })");
  const testing::DatasetForTesting dataset_false(feature_stats, false);

  EXPECT_FALSE(dataset_false.dataset_stats_view().by_weight());
  const testing::DatasetForTesting dataset_true(feature_stats, true);

  EXPECT_TRUE(dataset_true.dataset_stats_view().by_weight());
}

TEST(DatasetStatsView, GetByPathOrNull) {
  const testing::DatasetForTesting dataset(
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'optional_float'
        type: FLOAT
        num_stats: {
          common_stats: {
            num_missing: 0
            num_non_missing: 10
            min_num_values: 1
            max_num_values: 1
          }
        })"));
  EXPECT_EQ(
      "optional_float",
      dataset.dataset_stats_view().GetByPath(Path({"optional_float"}))->name());
  EXPECT_EQ(absl::nullopt,
            dataset.dataset_stats_view().GetByPath(Path({"imaginary_field"})));
}

TEST(DatasetStatsView, WeightedStatisticsExist) {
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 1000
        weighted_num_examples: 997.0
        features: {
          name: 'annotated_enum'
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 997
              max_num_values: 1
              weighted_common_stats: { num_missing: 0.0 num_non_missing: 997.0 }
            }
            unique: 3
            rank_histogram: { buckets: { label: "D" } }
          }
        })");
  EXPECT_TRUE(DatasetStatsView(statistics).WeightedStatisticsExist());
}

TEST(DatasetStatsView, WeightedStatisticsExistNoWeightedNumExamples) {
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 1000
        features: {
          name: 'annotated_enum'
          type: STRING
          string_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 997
              max_num_values: 1
              weighted_common_stats: { num_missing: 0.0 num_non_missing: 997.0 }
            }
            unique: 3
            rank_histogram: { buckets: { label: "D" } }
          }
        })");
  EXPECT_FALSE(DatasetStatsView(statistics).WeightedStatisticsExist());
}

TEST(DatasetStatsView, WeightedStatisticsExistNoWeightedCommonStats) {
  const DatasetFeatureStatistics statistics = ParseTextProtoOrDie<
      DatasetFeatureStatistics>(R"(
    num_examples: 1000
    weighted_num_examples: 997.0
    features: {
      name: 'annotated_enum'
      type: STRING
      string_stats: {
        common_stats: { num_missing: 3 num_non_missing: 997 max_num_values: 1 }
        unique: 3
        rank_histogram: { buckets: { label: "D" } }
      }
    })");
  EXPECT_FALSE(DatasetStatsView(statistics).WeightedStatisticsExist());
}

TEST(FeatureStatsView, GetParentGetPath) {
  const DatasetFeatureStatistics input =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: 'foo.baz'
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 6
              min_num_values: 3
              max_num_values: 7
            }
          }
        }
        features: {
          name: 'foo'
          type: STRUCT
          struct_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 6
              min_num_values: 1
              max_num_values: 1
            }
          }
        })");
  DatasetStatsView stats(input);

  absl::optional<FeatureStatsView> view = stats.GetByPath(Path({"foo", "baz"}));
  ASSERT_TRUE(view);
  absl::optional<FeatureStatsView> parent_view = view->GetParent();
  ASSERT_TRUE(parent_view);

  EXPECT_EQ(parent_view->name(), "foo");
  EXPECT_EQ(Path({"foo", "baz"}), view->GetPath());
}

TEST(DatasetStatsView, GetChildren) {
  const DatasetFeatureStatistics input =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: 'foo.bar'
          type: FLOAT
          num_stats: {
            common_stats: {
              num_missing: 3
              num_non_missing: 6
              min_num_values: 3
              max_num_values: 7
            }
          }
        }
        features: {
          name: 'foo'
          type: STRUCT
          struct_stats: {
            common_stats: {
              num_missing: 0
              num_non_missing: 6
              min_num_values: 1
              max_num_values: 1
            }
          }
        })");
  DatasetStatsView stats(input);
  absl::optional<FeatureStatsView> parent = stats.GetByPath(Path({"foo"}));
  ASSERT_TRUE(parent);
  std::vector<FeatureStatsView> children = parent->GetChildren();
  ASSERT_EQ(children.size(), 1);
  EXPECT_EQ(children[0].name(), "foo.bar");
}

}  // namespace data_validation
}  // namespace tensorflow
