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
#include "tensorflow_metadata/proto/v0/derived_feature.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {

using ::tensorflow::data_validation::testing::DatasetForTesting;
using ::tensorflow::data_validation::testing::ParseTextProtoOrDie;
using ::tensorflow::metadata::v0::DatasetFeatureStatistics;
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using ::testing::ContainerEq;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

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

  DatasetStatsView view(
      current, false, "environment_name", std::shared_ptr<DatasetStatsView>(),
      std::shared_ptr<DatasetStatsView>(), std::shared_ptr<DatasetStatsView>());

  EXPECT_TRUE(view.environment());
  EXPECT_EQ(*view.environment(), "environment_name");

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

  DatasetStatsView view(
      current, false, "environment_name", std::shared_ptr<DatasetStatsView>(),
      std::shared_ptr<DatasetStatsView>(), std::shared_ptr<DatasetStatsView>());

  EXPECT_TRUE(view.GetByPath(Path({"bar"}))->environment());
  EXPECT_EQ(*view.GetByPath(Path({"bar"}))->environment(), "environment_name");
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
                        std::shared_ptr<DatasetStatsView>(),
                        std::shared_ptr<DatasetStatsView>());

  const std::vector<std::pair<int, int>>
      view_by_path_previous_span_min_max_values = view.GetByPath(Path({"bar"}))
                                                      ->GetPreviousSpan()
                                                      ->GetMinMaxNumValues();
  ASSERT_EQ(view_by_path_previous_span_min_max_values.size(), 1);
  EXPECT_EQ(view_by_path_previous_span_min_max_values[0].first, 3);
  EXPECT_EQ(view_by_path_previous_span_min_max_values[0].second, 10);
  const std::vector<std::pair<int, int>> previous_span_by_path_min_max_values =
      view.GetPreviousSpan()->GetByPath(Path({"bar"}))->GetMinMaxNumValues();
  ASSERT_EQ(previous_span_by_path_min_max_values.size(), 1);
  EXPECT_EQ(previous_span_by_path_min_max_values[0].first, 3);
  EXPECT_EQ(previous_span_by_path_min_max_values[0].second, 10);
  const DatasetStatsView view_no_previous(current, false);
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
                        std::shared_ptr<DatasetStatsView>(), serving_view,
                        std::shared_ptr<DatasetStatsView>());

  const std::vector<std::pair<int, int>> view_serving_by_path_min_max_values =
      view.GetServing()->GetByPath(Path({"bar"}))->GetMinMaxNumValues();
  ASSERT_EQ(view_serving_by_path_min_max_values.size(), 1);
  EXPECT_EQ(view_serving_by_path_min_max_values[0].first, 3);
  EXPECT_EQ(view_serving_by_path_min_max_values[0].second, 10);
  const std::vector<std::pair<int, int>> view_by_path_serving_min_max_values =
      view.GetByPath(Path({"bar"}))->GetServing()->GetMinMaxNumValues();
  ASSERT_EQ(view_by_path_serving_min_max_values.size(), 1);
  EXPECT_EQ(view_by_path_serving_min_max_values[0].first, 3);
  EXPECT_EQ(view_by_path_serving_min_max_values[0].second, 10);
  const DatasetStatsView view_no_serving(current, false);
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

  EXPECT_EQ(DatasetForTesting(input).feature_stats_view().GetPath(),
            Path({"bar"}));
}

TEST(FeatureStatsView, Type) {
  const DatasetForTesting dataset(ParseTextProtoOrDie<FeatureNameStatistics>(R"(
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
  EXPECT_EQ(dataset.feature_stats_view().type(),
            tensorflow::metadata::v0::FeatureNameStatistics::FLOAT);
}

TEST(FeatureStatsView, DerivedSource) {
  const DatasetFeatureStatistics current =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"pb(
        features {
          name: 'foo'
          type: FLOAT
          num_stats: {}
        }
        features {
          name: 'bar'
          type: FLOAT
          validation_derived_source: { deriver_name: 'bar_source' }
          num_stats: {}
        }
      )pb");

  DatasetStatsView view(
      current, false, "environment_name", std::shared_ptr<DatasetStatsView>(),
      std::shared_ptr<DatasetStatsView>(), std::shared_ptr<DatasetStatsView>());
  EXPECT_FALSE(view.GetByPath(Path({"foo"}))->HasValidationDerivedSource());
  EXPECT_THAT(view.GetByPath(Path({"bar"}))->GetValidationDerivedSource(),
              testing::EqualsProto(
                  ParseTextProtoOrDie<metadata::v0::DerivedFeatureSource>(
                      "deriver_name: 'bar_source'")));
  EXPECT_TRUE(view.GetByPath(Path({"bar"}))->HasValidationDerivedSource());
}

TEST(FeatureStatsView, GetNumPresent) {
  const DatasetForTesting dataset(ParseTextProtoOrDie<FeatureNameStatistics>(R"(
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
  EXPECT_EQ(dataset.feature_stats_view().GetNumPresent(), 10.0);
}

TEST(FeatureStatsView, GetNumPresentForStruct) {
  const DatasetForTesting dataset(ParseTextProtoOrDie<FeatureNameStatistics>(R"(
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
  EXPECT_EQ(dataset.feature_stats_view().GetNumPresent(), 10.0);
}

TEST(FeatureStatsView, MinMaxNumValues) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: FLOAT
        num_stats: {
          common_stats: { num_missing: 3 min_num_values: 3 max_num_values: 7 }
        })");

  const std::vector<std::pair<int, int>> min_max_num_values =
      DatasetForTesting(input).feature_stats_view().GetMinMaxNumValues();
  ASSERT_EQ(min_max_num_values.size(), 1);
  EXPECT_EQ(min_max_num_values[0].first, 3);
  EXPECT_EQ(min_max_num_values[0].second, 7);
}

TEST(FeatureStatsView, MinMaxNumValuesMultipleNestednessLevels) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: FLOAT
        num_stats: {
          common_stats: {
            num_missing: 3
            min_num_values: 3
            max_num_values: 7
            presence_and_valency_stats: {
              num_missing: 3
              min_num_values: 3
              max_num_values: 7
            }
            presence_and_valency_stats: {
              num_missing: 3
              min_num_values: 4
              max_num_values: 8
            }
          }
        })");

  const std::vector<std::pair<int, int>> min_max_num_values =
      DatasetForTesting(input).feature_stats_view().GetMinMaxNumValues();
  EXPECT_THAT(min_max_num_values,
              ContainerEq(std::vector<std::pair<int, int>>{{3, 7}, {4, 8}}));
}

TEST(FeatureStatsView, GetNumMissingNested) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: FLOAT
        num_stats {
          common_stats {
            num_missing: 3
            weighted_common_stats { num_missing: 2 }
          }
        })");

  for (const auto by_weight_and_expected :
       std::vector<std::pair<bool, double>>{{true, 2}, {false, 3}}) {
    bool by_weight = by_weight_and_expected.first;
    double expected = by_weight_and_expected.second;

    auto got = DatasetForTesting(input, by_weight)
                   .feature_stats_view()
                   .GetNumMissingNested();

    ASSERT_EQ(got.size(), 1);
    EXPECT_EQ(got[0], expected);
  }
}

TEST(FeatureStatsView, GetNumMissingNestedMultipleNestednessLevels) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: FLOAT
        num_stats: {
          common_stats {
            num_missing: 3
            weighted_common_stats { num_missing: 2 }
            presence_and_valency_stats { num_missing: 2 }
            presence_and_valency_stats { num_missing: 0 }
            presence_and_valency_stats { num_missing: 1 }
            weighted_presence_and_valency_stats { num_missing: 3 }
            weighted_presence_and_valency_stats { num_missing: 0 }
            weighted_presence_and_valency_stats { num_missing: 2 }
          }

        })");

  for (const auto by_weight_and_expected :
       std::vector<std::pair<bool, std::vector<double>>>{{true, {3, 0, 2}},
                                                         {false, {2, 0, 1}}}) {
    bool by_weight = by_weight_and_expected.first;
    const auto& expected = by_weight_and_expected.second;

    auto got = DatasetForTesting(input, by_weight)
                   .feature_stats_view()
                   .GetNumMissingNested();

    EXPECT_EQ(got, expected);
  }
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
  EXPECT_EQ(DatasetForTesting(input).feature_stats_view().GetNumExamples(), 9);
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
  EXPECT_EQ(parent->GetPath(), Path({"foo"}));
  EXPECT_EQ(actual->GetPath(), Path({"foo", "bar"}));
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
  EXPECT_EQ(actual->GetPath(), Path({"foo.bar"}));
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
  EXPECT_EQ(roots[0].GetPath(), Path({"foo"}));
}

TEST(DatasetStatsView, GetRootFeaturesWithSkip) {
  const DatasetFeatureStatistics input =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: { name: 'bar' }
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
  EXPECT_EQ(roots[0].GetPath(), Path({"foo"}));
  EXPECT_TRUE(roots[0].GetChildren().empty());
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

  EXPECT_EQ(
      DatasetForTesting(input, false).feature_stats_view().GetNumExamples(),
      10.0);
  EXPECT_EQ(
      DatasetForTesting(input, true).dataset_stats_view().GetNumExamples(),
      8.5);
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

  const std::map<string, double> actual =
      DatasetForTesting(input).feature_stats_view().GetStringValuesWithCounts();
  EXPECT_THAT(actual, UnorderedElementsAre(Pair("foo", 1.5), Pair("bar", 2),
                                           Pair("baz", 3)));
}

TEST(FeatureStatsView, GetStringValuesWithCountsEmptyResult) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        num_stats: { common_stats: { num_missing: 3 max_num_values: 2 } })");

  const std::map<string, double> actual =
      DatasetForTesting(input).feature_stats_view().GetStringValuesWithCounts();
  EXPECT_THAT(actual, IsEmpty());
}

TEST(FeatureStatsView, GetStringValuesWithCountsNotString) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        num_stats: { common_stats: { num_non_missing: 4 num_missing: 1 } }
        type: STRING)");

  const std::map<string, double> actual =
      DatasetForTesting(input).feature_stats_view().GetStringValuesWithCounts();
  EXPECT_THAT(actual, IsEmpty());
}

TEST(FeatureStatsView, HasInvalidUTF8Strings) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: { num_missing: 3 max_num_values: 2 }
          unique: 3
          invalid_utf8_count: 1
          rank_histogram: {
            buckets: { label: "foo" sample_count: 1.5 }
            buckets: { label: "bar" sample_count: 2 }
            buckets: { label: "baz" sample_count: 3 }
          }
        })");

  EXPECT_TRUE(
      DatasetForTesting(input).feature_stats_view().HasInvalidUTF8Strings());
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

  EXPECT_FALSE(
      DatasetForTesting(input).feature_stats_view().HasInvalidUTF8Strings());
}

TEST(FeatureStatsView, GetStringValuesNoStringTest) {
  const tensorflow::metadata::v0::FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        num_stats: { common_stats: { num_non_missing: 4 num_missing: 1 } }
        type: STRING)");

  const std::vector<string> actual =
      DatasetForTesting(input).feature_stats_view().GetStringValues();
  EXPECT_THAT(actual, IsEmpty());
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
      DatasetForTesting(input).feature_stats_view().GetStringValues();
  EXPECT_THAT(actual, ElementsAre("bar", "baz", "foo"));
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
  const DatasetForTesting dataset(stats, true);
  const std::map<string, double> result =
      dataset.feature_stats_view().GetStringValuesWithCounts();
  EXPECT_THAT(result, ElementsAre(Pair("alpha", 123), Pair("beta", 234)));
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

  EXPECT_EQ(DatasetForTesting(input).feature_stats_view().num_stats().min(),
            8.0);
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
  EXPECT_EQ(DatasetForTesting(missing).feature_stats_view().num_stats().min(),
            0.0);
}

// Return the bytes stats, or an empty object if no bytes stats exist.
TEST(FeatureStatsView, BytesStats) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: BYTES
        bytes_stats: {
          avg_num_bytes: 10.0
          common_stats: { num_missing: 3 min_num_values: 3 max_num_values: 7 }
        })");

  EXPECT_EQ(DatasetForTesting(input)
                .feature_stats_view()
                .bytes_stats()
                .avg_num_bytes(),
            10.0);
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
  EXPECT_EQ(DatasetForTesting(missing).feature_stats_view().num_stats().min(),
            0.0);
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

  EXPECT_EQ(DatasetStatsView(input).features()[0].GetNumMissing(), 3.0);
  EXPECT_EQ(DatasetForTesting(input, true).feature_stats_view().GetNumMissing(),
            4.25);
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

  EXPECT_EQ(DatasetForTesting(input).feature_stats_view().GetFeatureType(),
            tensorflow::metadata::v0::STRUCT);
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

  absl::optional<double> result_a =
      DatasetForTesting(input).feature_stats_view().GetFractionPresent();
  EXPECT_EQ(*result_a, 0.75);
  absl::optional<double> result_b =
      DatasetForTesting(input, true).feature_stats_view().GetFractionPresent();
  EXPECT_EQ(*result_b, 0.25);
}

TEST(FeatureStatsView, GetTotalValueCountInExamples) {
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

  EXPECT_EQ(DatasetForTesting(input)
                .feature_stats_view()
                .GetTotalValueCountInExamples(),
            4.0);
}

// Older protos may not have tot_num_values set.
TEST(FeatureStatsView, GetTotalValueCountInExamplesOld) {
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

  EXPECT_EQ(DatasetForTesting(input)
                .feature_stats_view()
                .GetTotalValueCountInExamples(),
            4.0);
}

TEST(FeatureStatsView, GetNumUnique) {
  const FeatureNameStatistics categorical_input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: INT
        string_stats: {
          common_stats: {
            num_missing: 0
            num_non_missing: 2
            min_num_values: 1
            max_num_values: 1
          }
          unique: 2
        })");
  const FeatureNameStatistics string_input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: BYTES
        string_stats: {
          common_stats: {
            num_missing: 0
            num_non_missing: 2
            min_num_values: 1
            max_num_values: 2
          }
          unique: 3
        })");
  const FeatureNameStatistics numeric_input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: FLOAT
        num_stats: {
          common_stats: {
            num_missing: 0
            num_non_missing: 2
            min_num_values: 1
            max_num_values: 2
          }
        })");
  EXPECT_EQ(
      DatasetForTesting(categorical_input).feature_stats_view().GetNumUnique(),
      2);
  EXPECT_EQ(DatasetForTesting(string_input).feature_stats_view().GetNumUnique(),
            3);
  EXPECT_EQ(
      DatasetForTesting(numeric_input).feature_stats_view().GetNumUnique(),
      absl::nullopt);
}

TEST(FeatureStatsView, GetStandardHistogram) {
  const FeatureNameStatistics statistics_with_standard_histogram =
      ParseTextProtoOrDie<FeatureNameStatistics>(
          R"(name: 'float'
             type: FLOAT
             num_stats {
               common_stats { num_non_missing: 4 }
               histograms {
                 buckets { low_value: 1.0 high_value: 1.0 sample_count: 4.0 }
                 type: STANDARD
               }
             }
          )");
  const FeatureNameStatistics statistics_with_weighted_standard_histogram =
      ParseTextProtoOrDie<FeatureNameStatistics>(
          R"(
            name: 'integer'
            type: INT
            num_stats {
              common_stats {
                num_non_missing: 3
                weighted_common_stats { num_non_missing: 6.0 }
              }
              histograms {
                buckets { low_value: 2.0 high_value: 2.0 sample_count: 3.0 }
                type: STANDARD
              }
              weighted_numeric_stats {
                histograms {
                  buckets { low_value: 2.0 high_value: 2.0 sample_count: 6.0 }
                  type: STANDARD
                }
              }
            }
          )");
  const FeatureNameStatistics statistics_without_standard_histogram =
      ParseTextProtoOrDie<FeatureNameStatistics>(
          R"(name: 'float'
             type: FLOAT
             num_stats {
               common_stats { num_non_missing: 4 }
               histograms {
                 buckets { low_value: 1.0 high_value: 1.0 sample_count: 4.0 }
                 type: QUANTILES
               }
             }
          )");

  ASSERT_EQ(DatasetForTesting(statistics_with_standard_histogram)
                .feature_stats_view()
                .GetStandardHistogram()
                ->buckets_size(),
            1);
  tensorflow::metadata::v0::Histogram::Bucket actual_bucket =
      DatasetForTesting(statistics_with_standard_histogram)
          .feature_stats_view()
          .GetStandardHistogram()
          ->buckets()
          .at(0);
  EXPECT_EQ(actual_bucket.low_value(), 1);
  EXPECT_EQ(actual_bucket.high_value(), 1);
  EXPECT_EQ(actual_bucket.sample_count(), 4);
  ASSERT_EQ(DatasetForTesting(statistics_with_weighted_standard_histogram,
                              /*by_weight=*/true)
                .feature_stats_view()
                .GetStandardHistogram()
                ->buckets_size(),
            1);
  actual_bucket = DatasetForTesting(statistics_with_weighted_standard_histogram,
                                    /*by_weight=*/true)
                      .feature_stats_view()
                      .GetStandardHistogram()
                      ->buckets()
                      .at(0);
  EXPECT_EQ(actual_bucket.low_value(), 2);
  EXPECT_EQ(actual_bucket.high_value(), 2);
  EXPECT_EQ(actual_bucket.sample_count(), 6);

  EXPECT_EQ(DatasetForTesting(statistics_without_standard_histogram)
                .feature_stats_view()
                .GetStandardHistogram(),
            absl::nullopt);
}

// Return the custom stat by name, or an empty object if no custom stat exists.
TEST(FeatureStatsView, GetCustomStatsByName) {
  const FeatureNameStatistics input =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: FLOAT
        custom_stats: {
          name: "baz_custom"
          num: 8.0
        }
        num_stats: {
          min: 8.0
          common_stats: { num_missing: 3 min_num_values: 3 max_num_values: 7 }
        })");

  float expected_num = 8.0;
  EXPECT_EQ(DatasetForTesting(input)
                .feature_stats_view()
                .GetCustomStatByName("baz_custom")
                ->num(),
            expected_num);
  const FeatureNameStatistics missing =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: FLOAT
        custom_stats: { name: "bar_custom" num: 8.0 }
        num_stats: {
          min: 8.0
          common_stats: { num_missing: 3 min_num_values: 3 max_num_values: 7 }
        })");
  tensorflow::metadata::v0::CustomStatistic* expected = nullptr;
  EXPECT_EQ(DatasetForTesting(missing).feature_stats_view().GetCustomStatByName(
                "baz_custom"),
            expected);
}

TEST(DatasetStatsView, Features) {
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

  EXPECT_EQ(DatasetForTesting(input).dataset_stats_view().features().size(), 1);
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
  EXPECT_EQ(9, DatasetForTesting(input).dataset_stats_view().GetNumExamples());
}

TEST(DatasetStatsView, GetNumExamplesWeighted) {
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
  EXPECT_EQ(
      DatasetForTesting(input, true).dataset_stats_view().GetNumExamples(),
      8.5);
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
  const DatasetForTesting dataset_false(feature_stats, false);

  EXPECT_FALSE(dataset_false.dataset_stats_view().by_weight());
  const DatasetForTesting dataset_true(feature_stats, true);

  EXPECT_TRUE(dataset_true.dataset_stats_view().by_weight());
}

TEST(DatasetStatsView, GetByPathOrNull) {
  const DatasetForTesting dataset(ParseTextProtoOrDie<FeatureNameStatistics>(R"(
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
  EXPECT_EQ(dataset.dataset_stats_view()
                .GetByPath(Path({"optional_float"}))
                ->GetPath(),
            Path({"optional_float"}));
  EXPECT_EQ(dataset.dataset_stats_view().GetByPath(Path({"imaginary_field"})),
            absl::nullopt);
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

TEST(DatasetStatsView, WeightedStatisticsNoWeightedPresenceAndValency) {
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
              presence_and_valency_stats {
                num_missing: 3
                num_non_missing: 997
              }
              presence_and_valency_stats {
                num_non_missing: 1000
              }
            }
            unique: 3
            rank_histogram: { buckets: { label: "D" } }
          }
        })");
  EXPECT_FALSE(DatasetStatsView(statistics).WeightedStatisticsExist());
}

TEST(FeatureStatsView, GetParentGetPath_FieldIdIsName) {
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

  EXPECT_EQ(parent_view->GetPath(), Path({"foo"}));
  EXPECT_EQ(view->GetPath(), Path({"foo", "baz"}));
}

TEST(DatasetStatsView, GetChildren_FieldIdIsName) {
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
  EXPECT_EQ(children[0].GetPath(), Path({"foo", "bar"}));
}

TEST(FeatureStatsView, GetParentGetPath_FieldIdIsPath) {
  const DatasetFeatureStatistics input =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          path: {
            step: 'foo'
            step: 'baz'
          }
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
          path: {
            step: 'foo'
          }
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

  EXPECT_EQ(parent_view->GetPath(), Path({"foo"}));
  EXPECT_EQ(view->GetPath(), Path({"foo", "baz"}));
}

TEST(DatasetStatsView, GetChildren_FieldIdIsPath) {
  const DatasetFeatureStatistics input =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          path: {
            step: 'foo'
            step: 'bar'
          }
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
          path: {
            step: 'foo'
          }
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
  EXPECT_EQ(children[0].GetPath(), Path({"foo", "bar"}));
}

TEST(DatasetStatsViewDeathTest, MixedFieldId) {
  const DatasetFeatureStatistics input =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          path: { step: 'foo' step: 'bar' }
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
  EXPECT_DEATH({ DatasetStatsView stats(input); },
               "Some features had .name and some features had .path");
}

TEST(DatasetStatsView, GetPreviousVersion) {
  const DatasetFeatureStatistics current =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 1
        features {
          name: 'bar'
          type: FLOAT
          num_stats: {
            common_stats: { num_missing: 3 min_num_values: 3 max_num_values: 7 }
          }
        })");
  const DatasetFeatureStatistics previous_version = ParseTextProtoOrDie<
      DatasetFeatureStatistics>(R"(
    num_examples: 2
    features {
      name: 'bar'
      type: FLOAT
      num_stats: {
        common_stats: { num_missing: 3 min_num_values: 3 max_num_values: 10 }
      }
    })");

  std::shared_ptr<DatasetStatsView> previous_version_view =
      std::make_shared<DatasetStatsView>(previous_version);

  DatasetStatsView view(
      current, false, "environment_name", std::shared_ptr<DatasetStatsView>(),
      std::shared_ptr<DatasetStatsView>(), previous_version_view);

  std::vector<std::pair<int, int>> previous_min_max_num_values =
      view.GetPreviousVersion()->GetByPath(Path({"bar"}))->GetMinMaxNumValues();
  ASSERT_EQ(previous_min_max_num_values.size(), 1);
  EXPECT_EQ(previous_min_max_num_values[0].first, 3);
  EXPECT_EQ(previous_min_max_num_values[0].second, 10);
  DatasetStatsView view_no_previous_version(current, false);
  EXPECT_FALSE(view_no_previous_version.GetPreviousVersion());
}

}  // namespace data_validation
}  // namespace tensorflow
