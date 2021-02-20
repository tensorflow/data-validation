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

#include "tensorflow_data_validation/anomalies/feature_util.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow_data_validation/anomalies/statistics_view_test_util.h"
#include "tensorflow_data_validation/anomalies/test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {
namespace {
using ::tensorflow::metadata::v0::AnomalyInfo;
using ::tensorflow::metadata::v0::DatasetFeatureStatistics;
using ::tensorflow::metadata::v0::Feature;
using ::tensorflow::metadata::v0::FeatureComparator;
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using ::tensorflow::metadata::v0::FeaturePresence;
using ::tensorflow::metadata::v0::SparseFeature;
using ::tensorflow::metadata::v0::WeightedFeature;
using testing::AddWeightedStats;
using testing::EqualsProto;
using testing::ParseTextProtoOrDie;

std::string DescriptionsToString(const std::vector<Description>& descriptions) {
  std::string result;
  for (const Description& description : descriptions) {
    absl::StrAppend(&result, "short:", description.short_description, "\n");
    absl::StrAppend(&result, "long:", description.long_description, "\n");
  }
  return result;
}

Feature GetFeatureProtoOrDie(
    const tensorflow::metadata::v0::Schema& schema_proto,
    const std::string& field_name) {
  for (const Feature& feature_proto :
       schema_proto.feature()) {
    if (field_name == feature_proto.name()) {
      return feature_proto;
    }
  }
  LOG(FATAL) << "Name " << field_name << " not found in "
             << schema_proto.DebugString();
}

TEST(FeatureUtilTest, ClearDomain) {
  Feature feature = ParseTextProtoOrDie<Feature>(R"(
          name: "bytes_feature"
          presence: { min_count: 1 }
          value_count: { min: 1 max: 1 }
          type: BYTES
        )");
  EXPECT_EQ(feature.domain_info_case(), Feature::DOMAIN_INFO_NOT_SET);
  feature.mutable_int_domain();
  EXPECT_EQ(feature.domain_info_case(), Feature::kIntDomain);
  ClearDomain(&feature);
  EXPECT_EQ(feature.domain_info_case(), Feature::DOMAIN_INFO_NOT_SET);
}

struct LifecycleStageIsDeprecatedTest {
  metadata::v0::LifecycleStage stage;
  bool is_deprecated;
};

std::vector<LifecycleStageIsDeprecatedTest>
GetLifecycleStageIsDeprecatedTests() {
  return {{tensorflow::metadata::v0::DEPRECATED, true},
          {tensorflow::metadata::v0::DISABLED, true},
          {tensorflow::metadata::v0::ALPHA, true},
          {tensorflow::metadata::v0::PLANNED, true},
          {tensorflow::metadata::v0::DEBUG_ONLY, true},
          {tensorflow::metadata::v0::PRODUCTION, false},
          {tensorflow::metadata::v0::BETA, false},
          {tensorflow::metadata::v0::UNKNOWN_STAGE, false}};
}

TEST(FeatureUtilTest, LifecycleStageIsDeprecated) {
  for (const auto& test : GetLifecycleStageIsDeprecatedTests()) {
    EXPECT_EQ(LifecycleStageIsDeprecated(test.stage), test.is_deprecated)
        << "Failed on stage: " << test.stage << " expected is_deprecated: "
        << test.is_deprecated;
  }
}

TEST(FeatureUtilTest, FeatureIsDeprecatedLifecycleStage) {
  Feature feature;
  feature.set_lifecycle_stage(metadata::v0::DEPRECATED);
  EXPECT_TRUE(FeatureIsDeprecated(feature));
}

TEST(FeatureUtilTest, FeatureIsDeprecatedDeprecatedField) {
  Feature feature;
  feature.set_deprecated(true);
  feature.set_lifecycle_stage(metadata::v0::PRODUCTION);
  EXPECT_TRUE(FeatureIsDeprecated(feature));
}

TEST(FeatureUtilTest, FeatureIsDeprecatedEmpty) {
  Feature feature;
  EXPECT_FALSE(FeatureIsDeprecated(feature));
}

TEST(FeatureUtilTest, SparseFeatureIsDeprecatedLifecycleStage) {
  SparseFeature feature;
  feature.set_lifecycle_stage(metadata::v0::DEPRECATED);
  EXPECT_TRUE(SparseFeatureIsDeprecated(feature));
}

TEST(FeatureUtilTest, SparseFeatureIsDeprecatedDeprecatedField) {
  SparseFeature feature;
  feature.set_deprecated(true);
  feature.set_lifecycle_stage(metadata::v0::PRODUCTION);
  EXPECT_TRUE(SparseFeatureIsDeprecated(feature));
}

TEST(FeatureUtilTest, SparseFeatureIsDeprecatedEmpty) {
  SparseFeature feature;
  EXPECT_FALSE(SparseFeatureIsDeprecated(feature));
}

TEST(FeatureUtilTest, WeightedFeatureIsDeprecatedLifecycleStage) {
  WeightedFeature feature;
  feature.set_lifecycle_stage(metadata::v0::DEPRECATED);
  EXPECT_TRUE(WeightedFeatureIsDeprecated(feature));
}

TEST(FeatureUtilTest, WeightedFeatureIsDeprecatedEmpty) {
  WeightedFeature feature;
  EXPECT_FALSE(WeightedFeatureIsDeprecated(feature));
}

TEST(FeatureUtilTest, UpdateComparatorProposeNewThreshold) {
  DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 1
        features {
          name: "cat_feature"
          type: STRING
          string_stats: {
            rank_histogram {
              buckets {
                label: "tok1"
                sample_count: 10
              }
            }
          }
        }
        features {
          name: "num_feature"
          type: FLOAT
          num_stats {
            common_stats {}
            histograms {
              buckets { low_value: 0.0 high_value: 1.0 sample_count: 2.0 }
              buckets { low_value: 1.0 high_value: 2.0 sample_count: 2.0 }
              type: STANDARD
            }
          }
        })");
  DatasetFeatureStatistics previous_statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 1
        features {
          name: "cat_feature"
          type: STRING
          string_stats: {
            rank_histogram {
              buckets {
                label: "tok2"
                sample_count: 10
              }
            }
          }
        }
        features {
          name: "num_feature"
          type: INT
          num_stats {
            common_stats {}
            histograms {
              buckets { low_value: 3.0 high_value: 4.0 sample_count: 2.0 }
              buckets { low_value: 4.0 high_value: 6.0 sample_count: 2.0 }
              type: STANDARD
            }
          }
        })");
  // There are previous span statistics, but they do not contain stats for
  // 'test_feature'.
  DatasetStatsView stats_view(
      statistics, false, "environment_name",
      std::make_shared<DatasetStatsView>(previous_statistics),
      std::shared_ptr<DatasetStatsView>(), std::shared_ptr<DatasetStatsView>());

  {
    const FeatureStatsView feature_stats_view =
        stats_view.GetByPath(Path({"cat_feature"})).value();

    FeatureComparator comparator =
        ParseTextProtoOrDie<FeatureComparator>(R"(
          infinity_norm: { threshold: 0.1 })");

    auto result = UpdateFeatureComparatorDirect(
        feature_stats_view, FeatureComparatorType::DRIFT, &comparator);

    EXPECT_EQ(comparator.infinity_norm().threshold(), 1.0);
    ASSERT_EQ(result.measurements.size(), 1);
    EXPECT_THAT(result.measurements[0], EqualsProto(
        "type: L_INFTY value: 1 threshold: 0.1"));
  }
  {
    const FeatureStatsView feature_stats_view =
        stats_view.GetByPath(Path({"num_feature"})).value();
    FeatureComparator comparator = ParseTextProtoOrDie<FeatureComparator>(R"(
      jensen_shannon_divergence: { threshold: 0.1 })");
    auto result = UpdateFeatureComparatorDirect(
        feature_stats_view, FeatureComparatorType::DRIFT, &comparator);

    EXPECT_FLOAT_EQ(comparator.jensen_shannon_divergence().threshold(), 1.0);
    ASSERT_EQ(result.measurements.size(), 1);
    EXPECT_THAT(
        result.measurements[0],
        EqualsProto("type: JENSEN_SHANNON_DIVERGENCE value: 1 threshold: 0.1"));
  }
}

TEST(FeatureUtilTest,
     UpdateComparatorWithoutControlFeatureStatsClearsThreshold) {
  DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 1
        features {
          name: "test_feature"
          type: INT
          num_stats: { min: 0.0 max: 10.0 }
        }
        features {
          name: "other_feature"
          type: INT
          num_stats: { min: 0.0 max: 10.0 }
        })");
  DatasetFeatureStatistics previous_statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 1
        features {
          name: "other_feature"
          type: INT
          num_stats: { min: 0.0 max: 10.0 }
        })");
  // There are previous span statistics, but they do not contain stats for
  // 'test_feature'.
  DatasetStatsView stats_view(
      statistics, false, "environment_name",
      std::make_shared<DatasetStatsView>(previous_statistics),
      std::shared_ptr<DatasetStatsView>(), std::shared_ptr<DatasetStatsView>());
  const FeatureStatsView feature_stats_view =
      stats_view.GetByPath(Path({"test_feature"})).value();
  FeatureComparator comparator = ParseTextProtoOrDie<FeatureComparator>(R"(
    infinity_norm: { threshold: 0.1 })");

  auto result = UpdateFeatureComparatorDirect(
      feature_stats_view, FeatureComparatorType::DRIFT, &comparator);

  FeatureComparator expected_comparator =
      ParseTextProtoOrDie<FeatureComparator>(R"(infinity_norm: {})");
  EXPECT_THAT(comparator, EqualsProto(expected_comparator));
  // No comparison was done, thus no distance was measured.
  EXPECT_TRUE(result.measurements.empty());
}

TEST(FeatureUtilTest,
     UpdateComparatorWithoutControlFeatureStatsGeneratesAnomaly) {
  DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 1
        features {
          name: "test_feature"
          type: INT
          num_stats: { min: 0.0 max: 10.0 }
        }
        features {
          name: "other_feature"
          type: INT
          num_stats: { min: 0.0 max: 10.0 }
        })");
  DatasetFeatureStatistics previous_statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 1
        features {
          name: "other_feature"
          type: INT
          num_stats: { min: 0.0 max: 10.0 }
        })");
  // There are previous span statistics, but they do not contain stats for
  // 'test_feature'.
  DatasetStatsView stats_view(
      statistics, false, "environment_name",
      std::make_shared<DatasetStatsView>(previous_statistics),
      std::shared_ptr<DatasetStatsView>(), std::shared_ptr<DatasetStatsView>());
  const FeatureStatsView feature_stats_view =
      stats_view.GetByPath(Path({"test_feature"})).value();
  FeatureComparator comparator = ParseTextProtoOrDie<FeatureComparator>(R"(
    infinity_norm: { threshold: 0.1 }
    jensen_shannon_divergence: { threshold: 0.1 })");

  auto result = UpdateFeatureComparatorDirect(
      feature_stats_view, FeatureComparatorType::DRIFT, &comparator);

  const std::vector<Description>& actual_descriptions = result.descriptions;
  EXPECT_EQ(actual_descriptions.size(), 1);
  EXPECT_EQ(
      actual_descriptions[0].type,
      tensorflow::metadata::v0::AnomalyInfo::COMPARATOR_CONTROL_DATA_MISSING);
  EXPECT_EQ(actual_descriptions[0].short_description, "previous data missing");
  EXPECT_EQ(actual_descriptions[0].long_description,
            "previous data is missing.");
  // Confirm that missing control data clears the comparator threshold.
  EXPECT_FALSE(comparator.infinity_norm().has_threshold());
  EXPECT_FALSE(comparator.jensen_shannon_divergence().has_threshold());

  // No comparison was done, thus no distance was measured.
  EXPECT_TRUE(result.measurements.empty());
}

TEST(FeatureUtilTest,
     UpdateComparatorWithoutControlDatasetStatsMakesNoChanges) {
  DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        num_examples: 1
        features {
          name: "test_feature"
          type: INT
          num_stats: { min: 0.0 max: 10.0 }
        }
        features {
          name: "other_feature"
          type: INT
          num_stats: { min: 0.0 max: 10.0 }
        })");
  // There are no previous span statistics.
  DatasetStatsView stats_view(statistics);
  const FeatureStatsView feature_stats_view =
      stats_view.GetByPath(Path({"test_feature"})).value();
  FeatureComparator original_comparator =
      ParseTextProtoOrDie<FeatureComparator>(R"(
        infinity_norm: { threshold: 0.1 })");
  FeatureComparator comparator;
  comparator.CopyFrom(original_comparator);

  auto result = UpdateFeatureComparatorDirect(
      feature_stats_view, FeatureComparatorType::DRIFT, &comparator);
  const std::vector<Description>& actual_descriptions = result.descriptions;
  // The comparator is not changed, and no anomalies are generated.
  EXPECT_THAT(comparator, EqualsProto(original_comparator));
  EXPECT_EQ(actual_descriptions.size(), 0);

  // No comparison was done, thus no distance was measured.
  EXPECT_TRUE(result.measurements.empty());
}

TEST(FeatureUtilTest, UpdateUniqueConstraintsNoChange) {
  DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "categorical_feature"
          type: INT
          string_stats {
            common_stats: {
              num_missing: 0
              num_non_missing: 2
              min_num_values: 1
              max_num_values: 1
            }
            unique: 5
          }
        },
        features: {
          name: "string_feature"
          type: STRING
          string_stats {
            common_stats: {
              num_missing: 0
              num_non_missing: 2
              min_num_values: 1
              max_num_values: 1
            }
            unique: 1
          }
        })");

  DatasetStatsView stats_view(statistics);
  const FeatureStatsView categorical_feature_stats_view =
      stats_view.GetByPath(Path({"categorical_feature"})).value();
  const FeatureStatsView string_feature_stats_view =
      stats_view.GetByPath(Path({"string_feature"})).value();

  Feature original_categorical_feature =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Feature>(R"(
        name: "categorical_feature"
        type: INT
        int_domain { is_categorical: true }
        unique_constraints { min: 1 max: 5 })");
  Feature categorical_feature;
  categorical_feature.CopyFrom(original_categorical_feature);
  Feature original_string_feature =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Feature>(
          R"(name: "string_feature"
             type: BYTES
             unique_constraints { min: 1 max: 5 })");
  Feature string_feature;
  string_feature.CopyFrom(original_string_feature);

  std::vector<Description> actual_categorical_descriptions =
      UpdateUniqueConstraints(categorical_feature_stats_view,
                              &categorical_feature);
  std::vector<Description> actual_string_descriptions =
      UpdateUniqueConstraints(string_feature_stats_view, &string_feature);

  // The feature is not changed, and no anomalies are generated.
  EXPECT_THAT(categorical_feature, EqualsProto(original_categorical_feature));
  EXPECT_EQ(actual_categorical_descriptions.size(), 0);
  EXPECT_THAT(string_feature, EqualsProto(original_string_feature));
  EXPECT_EQ(actual_string_descriptions.size(), 0);
}

TEST(FeatureUtilTest, UpdateUniqueConstraintsNumUniquesOutsideRange) {
  DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "categorical_feature"
          type: INT
          string_stats {
            common_stats: {
              num_missing: 0
              num_non_missing: 2
              min_num_values: 1
              max_num_values: 1
            }
            unique: 5
          }
        },
        features: {
          name: "string_feature"
          type: STRING
          string_stats {
            common_stats: {
              num_missing: 0
              num_non_missing: 2
              min_num_values: 1
              max_num_values: 1
            }
            unique: 1
          }
        })");

  DatasetStatsView stats_view(statistics);
  const FeatureStatsView categorical_feature_stats_view =
      stats_view.GetByPath(Path({"categorical_feature"})).value();
  const FeatureStatsView string_feature_stats_view =
      stats_view.GetByPath(Path({"string_feature"})).value();

  Feature categorical_feature =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Feature>(R"(
        name: "categorical_feature"
        type: INT
        int_domain { is_categorical: true }
        unique_constraints { min: 2 max: 2 })");
  Feature string_feature =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Feature>(
          R"(name: "string_feature"
             type: BYTES
             unique_constraints { min: 2 max: 2 })");

  // The number of unique values for the categorical feature is higher than the
  // original unique_constraints.max for that feature, so expect that the max
  // will be updated.
  Feature expected_categorical_feature =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Feature>(R"(
        name: "categorical_feature"
        type: INT
        int_domain { is_categorical: true }
        unique_constraints { min: 2 max: 5 })");
  // The number of unique values for the string feature is lower than the
  // original unique_constraints.min for that feature, so expect that the
  // min will be updated.
  Feature expected_string_feature =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Feature>(
          R"(name: "string_feature"
             type: BYTES
             unique_constraints { min: 1 max: 2 })");

  std::vector<Description> actual_categorical_descriptions =
      UpdateUniqueConstraints(categorical_feature_stats_view,
                              &categorical_feature);
  std::vector<Description> actual_string_descriptions =
      UpdateUniqueConstraints(string_feature_stats_view, &string_feature);

  EXPECT_THAT(categorical_feature, EqualsProto(expected_categorical_feature));
  EXPECT_EQ(actual_categorical_descriptions.size(), 1);
  EXPECT_EQ(actual_categorical_descriptions.at(0).long_description,
            "Expected no more than 2 unique values but found 5.");
  EXPECT_THAT(string_feature, EqualsProto(expected_string_feature));
  EXPECT_EQ(actual_string_descriptions.size(), 1);
  EXPECT_EQ(actual_string_descriptions.at(0).long_description,
            "Expected at least 2 unique values but found only 1.");
}

TEST(FeatureUtilTest, UpdateUniqueConstraintsNotStringOrCategorical) {
  DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
        features: {
          name: "numeric_feature"
          type: INT
          num_stats {
            common_stats: {
              num_missing: 0
              num_non_missing: 6
              min_num_values: 1
              max_num_values: 1
            }
          }
        })");

  DatasetStatsView stats_view(statistics);
  const FeatureStatsView numeric_feature_stats_view =
      stats_view.GetByPath(Path({"numeric_feature"})).value();

  Feature numeric_feature =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Feature>(R"(
        name: "numeric_feature"
        type: INT
        unique_constraints { min: 5 max: 5 })");
  Feature expected_numeric_feature =
      ParseTextProtoOrDie<tensorflow::metadata::v0::Feature>(R"(
        name: "numeric_feature"
        type: INT)");

  std::vector<Description> actual_numeric_descriptions =
      UpdateUniqueConstraints(numeric_feature_stats_view, &numeric_feature);

  // The unique_constraints are cleared, and an anomaly is generated.
  EXPECT_THAT(numeric_feature, EqualsProto(expected_numeric_feature));
  EXPECT_EQ(actual_numeric_descriptions.size(), 1);
  EXPECT_EQ(actual_numeric_descriptions.at(0).long_description,
            "UniqueConstraints specified for the feature, but unique values "
            "were not counted (i.e., feature is not string or categorical).");
}

// Confirm that the result of calling DeprecateFeature on a feature is
// recognized as by FeatureIsDeprecated.
TEST(FeatureTypeTest, DeprecateConsistency) {
  Feature feature;
  feature.set_lifecycle_stage(metadata::v0::PRODUCTION);
  Feature deprecated = feature;
  DeprecateFeature(&deprecated);
  EXPECT_TRUE(FeatureIsDeprecated(deprecated))
      << "Failed to deprecate: " << feature.DebugString()
      << " produced " << deprecated.DebugString();
}

// Confirm that DeprecateFeature works on an empty feature.
TEST(FeatureTypeTest, DeprecateConsistencyEmpty) {
  Feature feature;
  Feature deprecated = feature;
  DeprecateFeature(&deprecated);
  EXPECT_TRUE(FeatureIsDeprecated(deprecated))
      << "Failed to deprecate: " << feature.DebugString()
      << " produced " << deprecated.DebugString();
}

// Construct a schema from a proto field, and then write it to a
// Feature.
struct FeatureNameStatisticsConstructorTest {
  FeatureNameStatistics statistics;
  Feature feature_proto;
  bool infer_shape;
};

// Repurpose for InitValueCountAndPresence.
// Also, break apart a separate test for other util constructors.
TEST(FeatureTypeTest, ConstructFromFeatureNameStatistics) {
  const std::vector<FeatureNameStatisticsConstructorTest> tests = {
      // Different min and max value counts with min above 0.
      {ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar1'
         type: STRING
         string_stats: {
           common_stats: {
             num_missing: 3
             num_non_missing: 10
             min_num_values: 1
             max_num_values: 5
           }
           unique: 3
           rank_histogram: {
             buckets: { label: "foo" }
             buckets: { label: "bar" }
             buckets: { label: "baz" }
           }
         })"),
       ParseTextProtoOrDie<Feature>(R"(
         value_count { min: 1 }
         presence { min_count: 1 }
       )"),
       /*infer_shape=*/false},
      // Different min and max value counts with min of 0.
      {ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar1'
         type: STRING
         string_stats: {
           common_stats: {
             num_missing: 3
             num_non_missing: 10
             min_num_values: 0
             max_num_values: 5
           }
           unique: 3
           rank_histogram: {
             buckets: { label: "foo" }
             buckets: { label: "bar" }
             buckets: { label: "baz" }
           }
         })"),
       ParseTextProtoOrDie<Feature>(R"(
         presence { min_count: 1 }
       )"),
       /*infer_shape=*/false},
      // Optional feature with same value count of 1.
      {ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar2'
         type: STRING
         string_stats: {
           common_stats: {
             num_missing: 3
             num_non_missing: 10
             min_num_values: 1
             max_num_values: 1
           }
           unique: 3
           rank_histogram: {
             buckets: { label: "foo" }
             buckets: { label: "bar" }
             buckets: { label: "baz" }
           }
         })"),
       ParseTextProtoOrDie<Feature>(R"(
         value_count { min: 1 max: 1 }
         presence { min_count: 1 }
       )"),
       /*infer_shape=*/false},
      // Optional feature with same value count above 1.
      {ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar3'
         type: INT
         num_stats: {
           common_stats: {
             num_missing: 3
             num_non_missing: 1
             max_num_values: 5
             min_num_values: 5
           }
         })"),
       ParseTextProtoOrDie<Feature>(R"(
         value_count { min: 5 max: 5 }
         presence { min_count: 1 }
       )"),
       /*infer_shape=*/false},
      // Required feature with same value count of 1.
      {ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         num_stats: {
           common_stats: {
             num_missing: 0
             num_non_missing: 1
             max_num_values: 1
             min_num_values: 1
           }
         })"),
       ParseTextProtoOrDie<Feature>(R"(
         value_count { min: 1 max: 1 }
         presence { min_count: 1 min_fraction: 1.0 }
       )"),
       /*infer_shape=*/false},
      // Required feature with same value count above 1.
      {ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar4'
         type: FLOAT
         num_stats: {
           common_stats: {
             num_missing: 0
             num_non_missing: 1
             max_num_values: 5
             min_num_values: 5
             weighted_common_stats: { num_missing: 0 num_non_missing: 0.5 }
           }
         })"),
       ParseTextProtoOrDie<Feature>(R"(
         value_count { min: 5 max: 5 }
         presence { min_count: 1 min_fraction: 1.0 }
       )"),
       /*infer_shape=*/false},
      {ParseTextProtoOrDie<FeatureNameStatistics>(
           R"(
             name: 'bar5'
             type: STRING
             string_stats: {
               common_stats: { num_missing: 100 num_non_missing: 0 }
             })"),
       ParseTextProtoOrDie<Feature>(R"(
         presence { min_count: 0 }
       )"),
       /*infer_shape=*/false},
      // shape not inferred at request because of presence.
      {ParseTextProtoOrDie<FeatureNameStatistics>(
           R"(
             name: 'shape_not_inferred_1'
             type: STRING
             string_stats: {
               common_stats: {
                 num_missing: 1
                 num_non_missing: 100
                 min_num_values: 1
                 max_num_values: 1
               }
             })"),
       ParseTextProtoOrDie<Feature>(R"(
         presence { min_count: 1 }
         value_count { min: 1 max: 1 }
       )"),
       /*infer_shape=*/true},
      // shape not inferred at request because of presence (nested this time).
      {ParseTextProtoOrDie<FeatureNameStatistics>(
           R"(
             name: 'shape_not_inferred_2'
             type: STRING
             string_stats: {
               common_stats: {
                 num_missing: 0
                 num_non_missing: 100
                 min_num_values: 1
                 max_num_values: 1
                 presence_and_valency_stats {
                   num_non_missing: 100
                   num_missing: 0
                   min_num_values: 1
                   max_num_values: 1
                 }
                 presence_and_valency_stats {
                   num_non_missing: 1000
                   num_missing: 1  # root cause
                   min_num_values: 2
                   max_num_values: 2
                 }
               }
             })"),
       ParseTextProtoOrDie<Feature>(R"(
         presence { min_count: 1 min_fraction: 1 }
         value_counts {
           value_count { min: 1 max: 1 }
           value_count { min: 2 max: 2 }
         }
       )"),
       /*infer_shape=*/true},
      // shape not inferred at request because of num_values.
      {ParseTextProtoOrDie<FeatureNameStatistics>(
           R"(
             name: 'shape_not_inferred_3'
             type: STRING
             string_stats: {
               common_stats: {
                 num_missing: 0
                 num_non_missing: 100
                 max_num_values: 1
               }
             })"),
       ParseTextProtoOrDie<Feature>(R"(
         presence { min_count: 1 min_fraction: 1 }
       )"),
       /*infer_shape=*/true},
      // shape not inferred at request because of num_values (nested this time).
      {ParseTextProtoOrDie<FeatureNameStatistics>(
           R"(
             name: 'shape_not_inferred_4'
             type: STRING
             string_stats: {
               common_stats: {
                 num_missing: 0
                 num_non_missing: 100
                 min_num_values: 1
                 max_num_values: 1
                 presence_and_valency_stats {
                   num_non_missing: 100
                   num_missing: 0
                   min_num_values: 1
                   max_num_values: 1
                 }
                 presence_and_valency_stats {
                   num_non_missing: 1000
                   num_missing: 0
                   min_num_values: 2
                   max_num_values: 4  # root cause
                 }
               }
             })"),
       ParseTextProtoOrDie<Feature>(R"(
         presence { min_count: 1 min_fraction: 1 }
         value_counts {
           value_count { min: 1 max: 1 }
           value_count { min: 1 }
         }
       )"),
       /*infer_shape=*/true},
      // shape inferred.
      {ParseTextProtoOrDie<FeatureNameStatistics>(
           R"(
             name: 'shape_inferred_1'
             type: STRING
             string_stats: {
               common_stats: {
                 num_missing: 0
                 num_non_missing: 100
                 min_num_values: 3
                 max_num_values: 3
               }
             })"),
       ParseTextProtoOrDie<Feature>(R"(
         presence { min_count: 1 min_fraction: 1 }
         shape { dim { size: 3 } }
       )"),
       /*infer_shape=*/true},
      // shape inferred (nested this time).
      {ParseTextProtoOrDie<FeatureNameStatistics>(
           R"(
             name: 'shape_inferred_2'
             type: STRING
             string_stats: {
               common_stats: {
                 num_missing: 0
                 num_non_missing: 100
                 min_num_values: 1
                 max_num_values: 1
                 presence_and_valency_stats {
                   num_non_missing: 100
                   num_missing: 0
                   min_num_values: 1
                   max_num_values: 1
                 }
                 presence_and_valency_stats {
                   num_non_missing: 1000
                   num_missing: 0
                   min_num_values: 2
                   max_num_values: 2
                 }
               }
             })"),
       ParseTextProtoOrDie<Feature>(R"(
         presence { min_count: 1 min_fraction: 1 }
         shape {
           dim { size: 1 }
           dim { size: 2 }
         }
       )"),
       /*infer_shape=*/true},
      // shape not inferred for STRUCT features.
      {ParseTextProtoOrDie<FeatureNameStatistics>(
           R"(
             name: "struct"
             type: STRUCT
             struct_stats: {
               common_stats: {
                 num_missing: 0
                 num_non_missing: 100
                 min_num_values: 1
                 max_num_values: 1
               }
             })"),
       ParseTextProtoOrDie<Feature>(R"(
         presence { min_count: 1 min_fraction: 1 }
         value_count { min: 1 max: 1 }
       )"),
       /*infer_shape=*/true},
  };
  for (const auto& test : tests) {
    Feature feature;

    const testing::DatasetForTesting dataset(AddWeightedStats(test.statistics),
                                             true);
    InitPresenceAndShape(dataset.feature_stats_view(), test.infer_shape,
                         &feature);
    EXPECT_THAT(feature, EqualsProto(test.feature_proto));
  }
}

struct UpdateFeatureValueCountsTest {
  std::string name;
  FeatureNameStatistics statistics;
  // Result of IsValid().
  bool expected_description_empty;
  std::unordered_set<tensorflow::metadata::v0::AnomalyInfo::Type>
      expected_anomaly_types;
  std::unordered_set<std::string> expected_short_descriptions;
  // Initial feature proto.
  Feature original;
  // Result of Update().
  Feature expected;
};

const std::vector<UpdateFeatureValueCountsTest>
GetUpdateFeatureValueCountsTests() {
  return {
      {"value_count_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'feature'
         type: FLOAT
         num_stats: {
           common_stats: {
             num_missing: 0
             num_non_missing: 10
             min_num_values: 1
             max_num_values: 1
           }
         })"),
       true,
       {},
       {},
       ParseTextProtoOrDie<Feature>(R"(value_count: { min: 1 max: 1 })"),
       ParseTextProtoOrDie<Feature>(R"(value_count: { min: 1 max: 1 })")},
      {"value_count_with_nestedness_mismatch",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'feature'
         type: FLOAT
         num_stats: {
           common_stats: {
             num_missing: 0
             num_non_missing: 10
             min_num_values: 1
             max_num_values: 1
             presence_and_valency_stats {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1
             }
             presence_and_valency_stats {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1
             }
           }
         })"),
       false,
       {AnomalyInfo::VALUE_NESTEDNESS_MISMATCH},
       {"Mismatched value nest level"},
       ParseTextProtoOrDie<Feature>(R"(value_count: { min: 1 max: 1 })"),
       ParseTextProtoOrDie<Feature>(R"(value_counts: {
                                         value_count: { min: 1 max: 1 }
                                         value_count: { min: 1 max: 1 }
                                       })")},
      {"num_values_outside_value_count_bounds",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'feature'
         type: FLOAT
         num_stats: {
           common_stats: {
             num_missing: 3
             num_non_missing: 10
             min_num_values: 3
             max_num_values: 8
           }
         })"),
       false,
       {AnomalyInfo::FEATURE_TYPE_LOW_NUMBER_VALUES,
        AnomalyInfo::FEATURE_TYPE_HIGH_NUMBER_VALUES},
       {"Missing values", "Superfluous values"},
       ParseTextProtoOrDie<Feature>(R"(value_count: { min: 5 max: 5 })"),
       ParseTextProtoOrDie<Feature>(R"(value_count: { min: 3 max: 8 })")},
      {"num_values_outside_value_count_bounds_clears_min",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'feature'
         type: FLOAT
         num_stats: {
           common_stats: {
             num_missing: 3
             num_non_missing: 10
             min_num_values: 0
             max_num_values: 8
           }
         })"),
       false,
       {AnomalyInfo::FEATURE_TYPE_LOW_NUMBER_VALUES,
        AnomalyInfo::FEATURE_TYPE_HIGH_NUMBER_VALUES},
       {"Missing values", "Superfluous values"},
       ParseTextProtoOrDie<Feature>(R"(value_count: { min: 5 max: 5 })"),
       ParseTextProtoOrDie<Feature>(R"(value_count: { max: 8 })")},
      {"value_counts_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'feature'
         type: FLOAT
         num_stats: {
           common_stats: {
             num_missing: 0
             num_non_missing: 10
             min_num_values: 1
             max_num_values: 1
             presence_and_valency_stats {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1
             }
             presence_and_valency_stats {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1
             }
           }
         })"),
       true,
       {},
       {},
       ParseTextProtoOrDie<Feature>(R"(value_counts: {
                                         value_count: { min: 1 max: 1 }
                                         value_count: { min: 1 max: 1 }
                                       })"),
       ParseTextProtoOrDie<Feature>(R"(value_counts: {
                                         value_count: { min: 1 max: 1 }
                                         value_count: { min: 1 max: 1 }
                                       })")},
      {
          "value_counts_nestedness_mismatch",
          ParseTextProtoOrDie<FeatureNameStatistics>(R"(
            name: 'feature'
            type: FLOAT
            num_stats: {
              common_stats: {
                num_missing: 0
                num_non_missing: 10
                min_num_values: 1
                max_num_values: 1
                presence_and_valency_stats {
                  num_missing: 0
                  num_non_missing: 10
                  min_num_values: 1
                  max_num_values: 1
                }
                presence_and_valency_stats {
                  num_missing: 0
                  num_non_missing: 10
                  min_num_values: 1
                  max_num_values: 1
                }
                presence_and_valency_stats {
                  num_missing: 0
                  num_non_missing: 10
                  min_num_values: 1
                  max_num_values: 1
                }
              }
            })"),
          false,
          {AnomalyInfo::VALUE_NESTEDNESS_MISMATCH},
          {"Mismatched value nest level"},
          ParseTextProtoOrDie<Feature>(R"(value_counts: {
                                            value_count: { min: 1 max: 1 }
                                            value_count: { min: 1 max: 1 }
                                          })"),
          ParseTextProtoOrDie<Feature>(R"(value_counts: {
                                            value_count: { min: 1 max: 1 }
                                            value_count: { min: 1 max: 1 }
                                            value_count: { min: 1 max: 1 }
                                          })"),
      },
      {"num_values_outside_value_counts_bounds",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'feature'
         type: FLOAT
         num_stats: {
           common_stats: {
             num_missing: 0
             num_non_missing: 10
             min_num_values: 1
             max_num_values: 1
             presence_and_valency_stats {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1
             }
             presence_and_valency_stats {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 3
               max_num_values: 8
             }
           }
         })"),
       false,
       {AnomalyInfo::FEATURE_TYPE_LOW_NUMBER_VALUES,
        AnomalyInfo::FEATURE_TYPE_HIGH_NUMBER_VALUES},
       {"Missing values", "Superfluous values"},
       ParseTextProtoOrDie<Feature>(R"(value_counts: {
                                         value_count: { min: 1 max: 1 }
                                         value_count: { min: 5 max: 5 }
                                       })"),
       ParseTextProtoOrDie<Feature>(R"(value_counts: {
                                         value_count: { min: 1 max: 1 }
                                         value_count: { min: 3 max: 8 }
                                       })")},
      {"num_values_outside_value_counts_bounds_clears_min",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'feature'
         type: FLOAT
         num_stats: {
           common_stats: {
             num_missing: 0
             num_non_missing: 10
             min_num_values: 1
             max_num_values: 1
             presence_and_valency_stats {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1
             }
             presence_and_valency_stats {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 0
               max_num_values: 8
             }
           }
         })"),
       false,
       {AnomalyInfo::FEATURE_TYPE_LOW_NUMBER_VALUES,
        AnomalyInfo::FEATURE_TYPE_HIGH_NUMBER_VALUES},
       {"Missing values", "Superfluous values"},
       ParseTextProtoOrDie<Feature>(R"(value_counts: {
                                         value_count: { min: 1 max: 1 }
                                         value_count: { min: 5 max: 5 }
                                       })"),
       ParseTextProtoOrDie<Feature>(R"(value_counts: {
                                         value_count: { min: 1 max: 1 }
                                         value_count: { max: 8 }
                                       })")},
  };
}

TEST(FeatureTypeTest, UpdateFeatureValueCountsTest) {
  for (const auto& test : GetUpdateFeatureValueCountsTests()) {
    Feature to_modify = test.original;
    testing::DatasetForTesting dataset(test.statistics);
    const std::vector<Description> description =
        UpdateFeatureValueCounts(dataset.feature_stats_view(), &to_modify);
    EXPECT_EQ(test.expected_description_empty, description.empty());
    if (!test.expected_description_empty) {
      std::unordered_set<AnomalyInfo::Type> actual_anomaly_types;
      std::unordered_set<std::string> actual_short_descriptions;
      for (const auto& each : description) {
        actual_anomaly_types.insert(each.type);
        actual_short_descriptions.insert(each.short_description);
      }
      EXPECT_EQ(actual_anomaly_types, test.expected_anomaly_types);
      EXPECT_EQ(actual_short_descriptions, test.expected_short_descriptions);
    }
    EXPECT_THAT(to_modify, EqualsProto(test.expected));
  }
}

// Construct a schema from a proto field, and then write it to a
// DescriptorProto.
struct UpdatePresenceTest {
  std::string name;
  FeatureNameStatistics statistics;
  bool expected_description_empty;
  // Initial feature proto.
  FeaturePresence original;
  // Result of Update().
  FeaturePresence expected;
};

// TODO(b/148430313): this is too many test cases.
// Wrap MinCountInvalid and MinCount tests into here.
// Remove redundant tests.
const std::vector<UpdatePresenceTest> GetUpdatePresenceTests() {
  const std::vector<UpdatePresenceTest> battery_a = {
      {"optional_float_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'optional_float'
           type: FLOAT
           num_stats: {
             common_stats: {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1}})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"optional_string_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'optional_string'
           type: STRING
           string_stats: {
             common_stats: {
               num_missing: 3
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 2
             }
             unique: 3
             rank_histogram: {
               buckets: {
                 label: "foo"
               }
               buckets: {
                 label: "bar"
               }
               buckets: {
                 label: "baz"}}})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"optional_int64_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'optional_int64'
           type: INT
           num_stats: {
             common_stats: {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1}})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"optional_bool_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'bar'
           type: INT
           num_stats: {
             common_stats: {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1
             }
             min: 0.0
             max: 1.0})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"repeated_float_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'repeated_float'
           type: FLOAT
           num_stats: {
             common_stats: {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1}})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"repeated_string_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'repeated_string'
           type: STRING
           string_stats: {
             common_stats: {
               num_missing: 3
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 2
             }
             unique: 3
             rank_histogram: {
               buckets: {
                 label: "foo"
               }
               buckets: {
                 label: "bar"
               }
               buckets: {
                 label: "baz"}}})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"repeated_int64_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'bar'
           type: INT
           num_stats: {
             common_stats: {
               num_missing: 10000
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1012
             }
             min: 0.0
             max: 1.0})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")}};
  const std::vector<UpdatePresenceTest> battery_b = {
      {"repeated_bool_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'bar'
           type: INT
           num_stats: {
             common_stats: {
               num_missing: 10000
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1012
             }
             min: 0.0
             max: 1.0})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"string_int64_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'repeated_string'
           type: STRING
           string_stats: {
             common_stats: {
               num_missing: 3
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1
             }
             unique: 3
             rank_histogram: {
               buckets: {
                 label: "12"
               }
               buckets: {
                 label: "39"
               }
               buckets: {
                 label: "256"
               }
             }})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"string_int64_to_repeated",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'repeated_string'
           type: STRING
           string_stats: {
             common_stats: {
               num_missing: 3
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 2
             }
             unique: 3
             rank_histogram: {
               buckets: {
                 label: "12"
               }
               buckets: {
                 label: "39"
               }
               buckets: {
                 label: "256"}}})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {
          "string_int32_valid",
          ParseTextProtoOrDie<FeatureNameStatistics>(R"(
              name: 'string_int32'
              type: STRING
              string_stats: {
                common_stats: {
                  num_missing: 3
                  num_non_missing: 10
                  min_num_values: 1
                  max_num_values: 1
                }
                unique: 3
                rank_histogram: {
                  buckets: {
                    label: "12"
                  }
                  buckets: {
                    label: "39"
                  }
                  buckets: {
                    label: "256"}}})"),
          true,
          ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
          ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
      },
      {"string_int32_to_repeated_string",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'string_int32'
           type: STRING
           string_stats: {
             common_stats: {
               num_missing: 3
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 2
             }
             unique: 3
             rank_histogram: {
               buckets: {
                 label: "FOO"
               }
               buckets: {
                 label: "39"
               }
               buckets: {
                 label: "256"}}})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"string_uint32_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'string_uint32'
           type: STRING
           string_stats: {
             common_stats: {
               num_missing: 3
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1
             }
             unique: 3
             rank_histogram: {
               buckets: {
                 label: "12"
               }
               buckets: {
                 label: "39"
               }
               buckets: {
                 label: "256"}}})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")}};
  const std::vector<UpdatePresenceTest> battery_c = {
      {"string_uint32_negatives",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'string_uint32'
           type: STRING
           string_stats: {
             common_stats: {
               num_missing: 3
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 1
             }
             unique: 3
             rank_histogram: {
               buckets: {
                 label: "-12"
               }
               buckets: {
                 label: "39"
               }
               buckets: {
                 label: "256"}}})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"few_int64_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'bar'
           type: INT
           num_stats: {
             common_stats: {
               num_missing: 0
               num_non_missing: 10
               min_num_values: 1
               max_num_values: 3
             }
             min: 0.0
             max: 1.0})"),
       true, ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1)")},
      {"float_very_common_valid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'float_very_common'
           type: FLOAT
           num_stats: {
             common_stats: {
               num_missing: 4
               num_non_missing: 6
               min_num_values: 1
               max_num_values: 1
             }
             min: 0.0
             max: 1.0})"),
       true,
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1 min_fraction: 0.5)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1 min_fraction: 0.5)")},
      {"float_very_common_invalid",
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
           name: 'float_very_common'
           type: FLOAT
           num_stats: {
             common_stats: {
               num_missing: 7  # 7/10 missing.
               num_non_missing: 3
               min_num_values: 1
               max_num_values: 1
               weighted_common_stats: {
                 num_non_missing: 3.0
                 num_missing: 7.0
               }
             }
             min: 0.0
             max: 1.0})"),
       false,
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1 min_fraction: 0.5)"),
       ParseTextProtoOrDie<FeaturePresence>(R"(min_count: 1 min_fraction: 0.3)")}};
  std::vector<UpdatePresenceTest> result(battery_a);
  result.insert(result.end(), battery_b.begin(), battery_b.end());
  result.insert(result.end(), battery_c.begin(), battery_c.end());
  return result;
}

TEST(FeatureTypeTest, UpdatePresenceTest) {
  for (const auto& test : GetUpdatePresenceTests()) {
    for (bool by_weight : {false, true}) {
      FeaturePresence to_modify = test.original;
      DatasetFeatureStatistics statistics;
      statistics.set_num_examples(10);
      *statistics.add_features() = test.statistics;
      testing::DatasetForTesting dataset(AddWeightedStats(test.statistics),
                                         by_weight);
      std::vector<Description> description_b;
      description_b = UpdatePresence(dataset.feature_stats_view(), &to_modify);
      EXPECT_EQ(test.expected_description_empty, description_b.empty());
      EXPECT_THAT(to_modify, EqualsProto(test.expected))
          << "Test:" << test.name << "(by_weight: " << by_weight
          << ") Reason: " << DescriptionsToString(description_b);
    }
  }
}


// Construct a schema from a proto field, and then write it to a
// DescriptorProto.
struct UpdateShapeTestCase {
  std::string name;
  FeatureNameStatistics statistics;
  // Initial feature proto.
  Feature original;
  // Result of Update().
  Feature expected;
  bool expected_description_empty;
  bool generate_legacy_feature_spec;
};

const std::vector<UpdateShapeTestCase> GetUpdateShapeTestCases() {
  return {{
              "no shape",
              ParseTextProtoOrDie<FeatureNameStatistics>(R"(
                name: 'f1'
                type: INT
                num_stats: {
                  common_stats: {
                    num_missing: 0
                    num_non_missing: 10
                    min_num_values: 1
                    max_num_values: 1
                  }
                }
              )"),
              Feature(),
              Feature(),
              /*expected_description_empty=*/true,
              /*generate_legacy_feature_spec=*/false,
          },
          {
              "validation passes",
              ParseTextProtoOrDie<FeatureNameStatistics>(R"(
                name: 'f1'
                type: INT
                num_stats: {
                  common_stats: {
                    num_missing: 0
                    num_non_missing: 10
                    min_num_values: 1
                    max_num_values: 1
                  }
                }
              )"),
              ParseTextProtoOrDie<Feature>(R"(shape { dim { size: 1 } })"),
              ParseTextProtoOrDie<Feature>(R"(shape { dim { size: 1 } })"),
              /*expected_description_empty=*/true,
              /*generate_legacy_feature_spec=*/false,
          },
          {
              "validation passes: scalar",
              ParseTextProtoOrDie<FeatureNameStatistics>(R"(
                name: 'f1'
                type: INT
                num_stats: {
                  common_stats: {
                    num_missing: 0
                    num_non_missing: 10
                    min_num_values: 1
                    max_num_values: 1
                  }
                }
              )"),
              ParseTextProtoOrDie<Feature>(R"(shape {})"),
              ParseTextProtoOrDie<Feature>(R"(shape {})"),
              /*expected_description_empty=*/true,
              /*generate_legacy_feature_spec=*/false,
          },
          {
              "validation passes: fancy shape",
              ParseTextProtoOrDie<FeatureNameStatistics>(R"(
                name: 'f1'
                type: INT
                num_stats: {
                  common_stats: {
                    num_missing: 0
                    num_non_missing: 10
                    min_num_values: 36
                    max_num_values: 36
                  }
                }
              )"),
              ParseTextProtoOrDie<Feature>(R"(shape {
                                                dim { size: 2 }
                                                dim { size: 2 },
                                                dim { size: 9 }
                                              })"),
              ParseTextProtoOrDie<Feature>(R"(shape {
                                                dim { size: 2 }
                                                dim { size: 2 },
                                                dim { size: 9 }
                                              })"),
              /*expected_description_empty=*/true,
              /*generate_legacy_feature_spec=*/false,
          },
          {
              "validation passes: fancy shape, nested",
              ParseTextProtoOrDie<FeatureNameStatistics>(R"(
                name: 'f1'
                type: INT
                num_stats: {
                  common_stats: {
                    num_missing: 0
                    num_non_missing: 10
                    min_num_values: 2
                    max_num_values: 2
                    # 2-nested list, of shape [2, 18]. It's compatible with (can
                    # be reshaped to) [2, 2, 9].
                    presence_and_valency_stats {
                      num_missing: 0
                      num_non_missing: 10
                      min_num_values: 2
                      max_num_values: 2
                    }
                    presence_and_valency_stats {
                      num_missing: 0
                      num_non_missing: 20
                      min_num_values: 18
                      max_num_values: 18
                    }
                  }

                }
              )"),
              ParseTextProtoOrDie<Feature>(R"(shape {
                                                dim { size: 2 }
                                                dim { size: 2 },
                                                dim { size: 9 }
                                              })"),
              ParseTextProtoOrDie<Feature>(R"(shape {
                                                dim { size: 2 }
                                                dim { size: 2 },
                                                dim { size: 9 }
                                              })"),
              /*expected_description_empty=*/true,
              /*generate_legacy_feature_spec=*/false,
          },
          {
              "failure: num_missing",
              ParseTextProtoOrDie<FeatureNameStatistics>(R"(
                name: 'f1'
                type: INT
                num_stats: {
                  common_stats: {
                    num_missing: 1
                    num_non_missing: 10
                    min_num_values: 1
                    max_num_values: 1
                  }
                }
              )"),
              ParseTextProtoOrDie<Feature>(R"(shape { dim { size: 1 } })"),
              Feature(),
              /*expected_description_empty=*/false,
              /*generate_legacy_feature_spec=*/false,
          },
          {
              "failure: num_missing (nested)",
              ParseTextProtoOrDie<FeatureNameStatistics>(R"(
                name: 'f1'
                type: INT
                num_stats: {
                  common_stats: {
                    num_missing: 0
                    num_non_missing: 10
                    min_num_values: 1
                    max_num_values: 1
                    presence_and_valency_stats {
                      num_missing: 0
                      num_non_missing: 10
                      min_num_values: 1
                      max_num_values: 1
                    }
                    presence_and_valency_stats {
                      num_missing: 1
                      num_non_missing: 9
                      min_num_values: 1
                      max_num_values: 1
                    }
                  }
                }
              )"),
              ParseTextProtoOrDie<Feature>(R"(shape { dim { size: 1 } })"),
              Feature(),
              /*expected_description_empty=*/false,
              /*generate_legacy_feature_spec=*/false,
          },
          {
              "failure: num_value (nested)",
              ParseTextProtoOrDie<FeatureNameStatistics>(R"(
                name: 'f1'
                type: INT
                num_stats: {
                  common_stats: {
                    num_missing: 0
                    num_non_missing: 10
                    min_num_values: 1
                    max_num_values: 1
                    presence_and_valency_stats {
                      num_missing: 0
                      num_non_missing: 10
                      min_num_values: 1
                      max_num_values: 1
                    }
                    presence_and_valency_stats {
                      num_missing: 0
                      num_non_missing: 9
                      min_num_values: 0  # root cause
                      max_num_values: 1
                    }
                  }
                }
              )"),
              ParseTextProtoOrDie<Feature>(R"(shape { dim { size: 1 } })"),
              Feature(),
              /*expected_description_empty=*/false,
              /*generate_legacy_feature_spec=*/false,
          },
          {
              "failure: shape not compatible",
              ParseTextProtoOrDie<FeatureNameStatistics>(R"(
                name: 'f1'
                type: INT
                num_stats: {
                  common_stats: {
                    num_missing: 0
                    num_non_missing: 10
                    min_num_values: 1
                    max_num_values: 1
                  }
                })"),
              ParseTextProtoOrDie<Feature>(R"(shape {
                                                dim { size: 2 }
                                                dim { size: 3 }
                                              })"),
              Feature(),
              /*expected_description_empty=*/false,
              /*generate_legacy_feature_spec=*/false,
          },
          {
              "success: num_missing but generate_legacy_feature_spec",
              ParseTextProtoOrDie<FeatureNameStatistics>(R"(
                name: 'f1'
                type: INT
                num_stats: {
                  common_stats: {
                    num_missing: 1
                    num_non_missing: 10
                    min_num_values: 1
                    max_num_values: 1
                  }
                }
              )"),
              ParseTextProtoOrDie<Feature>(R"(shape { dim { size: 1 } })"),
              ParseTextProtoOrDie<Feature>(R"(shape { dim { size: 1 } })"),
              /*expected_description_empty=*/true,
              /*generate_legacy_feature_spec=*/true,
          },
         };
}

TEST(FeatureTypeTest, UpdateShapeTest) {
  for (const auto& test : GetUpdateShapeTestCases()) {
    for (const bool by_weight : {false, true}) {
      DatasetFeatureStatistics statistics;
      statistics.set_num_examples(10);
      *statistics.add_features() = test.statistics;
      testing::DatasetForTesting dataset(AddWeightedStats(test.statistics),
                                         by_weight);
      Feature updated = test.original;
      auto descriptions =
          UpdateFeatureShape(dataset.feature_stats_view(),
                             test.generate_legacy_feature_spec, &updated);
      EXPECT_EQ(test.expected_description_empty, descriptions.empty())
          << "Test: " << test.name;
      EXPECT_THAT(updated, EqualsProto(test.expected))
          << "Test:" << test.name << "(by_weight: " << by_weight
          << ") Reason: " << DescriptionsToString(descriptions);
    }
  }
}

TEST(FeatureTypeTest, RareMissingFeatureUnweighted) {
  // Rare feature missing for unweighted stats.
  // Notice that the feature is not technically missing given num_non_missing,
  DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
          num_examples:1000000000000000
          features {
            name: "one_in_a_quadrillion"
            type:INT
            num_stats: {
              common_stats: {
                num_missing: 1
                num_non_missing: 1000000000000000
                min_num_values: 1
                max_num_values: 1
              }
              min: 0.0
              max: 10.0
            }
          })");
  DatasetStatsView stats_view(statistics);
  const absl::optional<FeatureStatsView> feature_stats_view =
      stats_view.GetByPath(Path({"one_in_a_quadrillion"}));
  Feature to_modify = ParseTextProtoOrDie<Feature>(R"(
      name: "one_in_a_quadrillion"
      presence {
        min_fraction: 1.0
      }
      type: INT
      value_count {
        min: 1
        max: 1
      }
  )");
  std::vector<Description> desc =
      UpdatePresence(*feature_stats_view, to_modify.mutable_presence());
  ASSERT_EQ(desc.size(), 1);
  EXPECT_EQ(
      desc.at(0).long_description,
      "The feature was expected everywhere, but was missing in 1 examples.");
}

TEST(FeatureTypeTest, RareMissingFeatureWeighted) {
  // Rare feature missing for unweighted stats.
  // Notice that the feature is not technically missing given num_non_missing,
  DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
          num_examples:1000000000000000
          weighted_num_examples: 1000000000000000
          features {
            name: "one_in_a_quadrillion"
            type:INT
            num_stats: {
              common_stats: {
                num_missing: 1
                num_non_missing: 1000000000000000
                min_num_values: 1
                max_num_values: 1
                weighted_common_stats {
                  num_non_missing: 1000000000000000
                  num_missing: 1
                }
              }
              min: 0.0
              max: 10.0
            }
          })");
  DatasetStatsView stats_view(statistics, true);
  const absl::optional<FeatureStatsView> feature_stats_view =
      stats_view.GetByPath(Path({"one_in_a_quadrillion"}));
  Feature to_modify = ParseTextProtoOrDie<Feature>(R"(
      name: "one_in_a_quadrillion"
      presence {
        min_fraction: 1.0
      }
      type: INT
      value_count {
        min: 1
        max: 1
      }
  )");
  std::vector<Description> desc =
      UpdatePresence(*feature_stats_view, to_modify.mutable_presence());
  ASSERT_EQ(desc.size(), 1);
  EXPECT_EQ(
      desc.at(0).long_description,
      "The feature was expected everywhere, but was missing in 1 examples.");
}

// Tests IsValid and Update.
TEST(FeatureTypeTest, TestMinCount) {
  // Implicitly cardinality: CUSTOM.
  const Feature feature_proto =
      ParseTextProtoOrDie<Feature>(
          "name: 'foo' presence:{min_count: 3} type:INT");
  Feature to_modify = feature_proto;
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
          num_examples:10
          weighted_num_examples: 10.0
          features {
            name: 'foo'
            type: INT
            num_stats: {
              common_stats: {
                num_missing: 4
                num_non_missing: 6
                min_num_values: 1
                max_num_values: 1
                weighted_common_stats: {
                  num_non_missing: 6.0
                }
              }
              min: 0.0
              max: 1.0
            }})");
  const testing::DatasetForTesting dataset(statistics, false);
  std::vector<Description> description_b;
  if (to_modify.has_presence()) {
    description_b = UpdatePresence(dataset.feature_stats_view(),
                                   to_modify.mutable_presence());
  }
  EXPECT_TRUE(description_b.empty());
  EXPECT_THAT(to_modify, EqualsProto(feature_proto));
}

// Tests IsValid and Update.
TEST(FeatureTypeTest, TestMinCountInvalid) {
  const Feature feature_proto =
      ParseTextProtoOrDie<Feature>(
          "name: 'foo' presence:{min_count: 7} type:INT");
  Feature to_modify = feature_proto;
  const DatasetFeatureStatistics statistics =
      ParseTextProtoOrDie<DatasetFeatureStatistics>(R"(
          num_examples:10
          features {
            name: 'foo'
            type: INT
            num_stats: {
              common_stats: {
                num_missing: 4
                num_non_missing: 6
                min_num_values: 1
                max_num_values: 1
              }
              min: 0.0
              max: 1.0}})");
  const testing::DatasetForTesting dataset(statistics, false);
  std::vector<Description> description_b;
  if (to_modify.has_presence()) {
    description_b = UpdatePresence(dataset.feature_stats_view(),
                                   to_modify.mutable_presence());
  }
  EXPECT_FALSE(description_b.empty());
  EXPECT_THAT(to_modify, EqualsProto(R"(
      name: "foo"
      type: INT
      presence {
        min_count: 6})"));
}

struct UpdateMaxCadinalityConstraintTest {
  // Description of the feature.
  Feature feature_proto;
  // If true then no messages should be generated as part of the update.
  bool empty_messages;
  std::string description;
};

TEST(FeatureTypeTest, ValidateCustomFeatures) {
  const FeatureNameStatistics feature_stats =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'foo'
        type: STRING
        string_stats: {
          common_stats: {
            num_missing: 3
            num_non_missing: 7
            min_num_values: 1
            max_num_values: 2
            weighted_common_stats: {
              num_non_missing: 7.0
            }
          }
          weighted_string_stats: {
          }
        })");

  const std::vector<UpdateMaxCadinalityConstraintTest> tests = {
      {ParseTextProtoOrDie<Feature>(R"(name: "foo"
                           value_count {
                             min: 1
                           }
                           type: BYTES
                           presence {
                             min_count: 1
                           })"),
       true, "Custom feature should have empty messages"},
      {ParseTextProtoOrDie<Feature>(R"(name: "foo"
                           value_count {
                             min: 1
                           }
                           type: BYTES
                           presence {
                             min_count: 1
                           })"),
       true, "Repeated feature should have empty messages"},
      {ParseTextProtoOrDie<Feature>(R"(name: "foo"
                           value_count {
                             min: 1
                             max: 1
                           }
                           type: BYTES
                           presence {
                             min_count: 1
                           })"),
       false, "Optional feature should have some messages"},
      {ParseTextProtoOrDie<Feature>(R"(name: "foo"
                           value_count {
                             min: 1
                             max: 1
                           }
                           type: BYTES
                           presence {
                             min_count: 1
                             min_fraction: 1
                           })"),
       false, "Required feature should have some messages"}};
  for (const auto& test : tests) {
    for (bool by_weight : {true, false}) {
      const testing::DatasetForTesting dataset(feature_stats, false);
      Feature to_modify = test.feature_proto;

      std::vector<Description> description_a;
      if (to_modify.has_value_count()) {
        description_a =
            UpdateFeatureValueCounts(dataset.feature_stats_view(), &to_modify);
      }
      std::vector<Description> description_b;
      if (to_modify.has_presence()) {
        description_b = UpdatePresence(dataset.feature_stats_view(),
                                       to_modify.mutable_presence());
      }

      EXPECT_EQ(test.empty_messages,
                description_a.empty() && description_b.empty())
          << test.description << "(by_weight: " << by_weight << ")";
    }
  }
}

TEST(FeatureTypeTest, HasSkewComparatorFalse) {
  const Feature feature = ParseTextProtoOrDie<Feature>(R"(name: "feature_name")");
  EXPECT_FALSE(FeatureHasComparator(feature, FeatureComparatorType::SKEW));
}

TEST(FeatureTypeTest, HasSkewComparatorTrue) {
  const Feature feature = ParseTextProtoOrDie<Feature>(R"(name: "feature_name"
    skew_comparator {})");
  EXPECT_TRUE(FeatureHasComparator(feature, FeatureComparatorType::SKEW));
}

TEST(FeatureTypeTest, MutableSkewComparator) {
  Feature feature = ParseTextProtoOrDie<Feature>(R"(
      name: "feature_name"
      skew_comparator: {
        infinity_norm: {
          threshold: 0.1}})");
  FeatureComparator* comparator =
      GetFeatureComparator(&feature, FeatureComparatorType::SKEW);
  ASSERT_TRUE(comparator != nullptr);
  EXPECT_THAT(*comparator, EqualsProto(R"(
    infinity_norm: { threshold: 0.1 })"));
}

TEST(FeatureTypeTest, MutableComparator2) {
  Feature feature = ParseTextProtoOrDie<Feature>(R"(
      name: "feature_name"
      skew_comparator: {
        infinity_norm: {
        threshold: 0.1}})");
  FeatureComparator* comparator =
      GetFeatureComparator(&feature, FeatureComparatorType::SKEW);
  ASSERT_TRUE(comparator != nullptr);
  comparator->mutable_infinity_norm()->set_threshold(0.2);
  EXPECT_THAT(feature.skew_comparator(),
              EqualsProto(R"(infinity_norm: { threshold: 0.2 })"));
}

TEST(FeatureTypeTest, MutableComparatorWithDrift) {
  Feature feature = ParseTextProtoOrDie<Feature>(R"(
      name: "feature_name"
      drift_comparator: {
        infinity_norm: {
        threshold: 0.1}})");
  FeatureComparator* comparator =
      GetFeatureComparator(&feature, FeatureComparatorType::DRIFT);
  ASSERT_TRUE(comparator != nullptr);
  comparator->mutable_infinity_norm()->set_threshold(0.2);
  EXPECT_THAT(feature.drift_comparator(),
              EqualsProto(R"(infinity_norm: { threshold: 0.2 })"));
}

TEST(FeatureTypeTest, GetComparatorNormal) {
  Feature feature = ParseTextProtoOrDie<Feature>(R"(
      name: "feature_name"
      skew_comparator: {
        infinity_norm: {
          threshold: 0.1}})");
  FeatureComparator* comparator =
      GetFeatureComparator(&feature, FeatureComparatorType::SKEW);

  ASSERT_TRUE(comparator != nullptr);
  EXPECT_THAT(*comparator, EqualsProto("infinity_norm: { threshold: 0.1 }"));
}

}  // namespace
}  // namespace data_validation
}  // namespace tensorflow
