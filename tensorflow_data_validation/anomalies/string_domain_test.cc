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

// TODO(b/148430551): These tests are kept as they were earlier, to ensure that
// there is no change in logic. Eventually, they should be renamed and modified
// to test the current API better.
#include "tensorflow_data_validation/anomalies/string_domain_util.h"

#include <stddef.h>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow_data_validation/anomalies/proto/feature_statistics_to_proto.pb.h"
#include "tensorflow_data_validation/anomalies/statistics_view_test_util.h"
#include "tensorflow_data_validation/anomalies/test_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {
namespace {

const int64 kDefaultEnumThreshold = 400;

FeatureStatisticsToProtoConfig GetDefaultFeatureStatisticsToProtoConfig() {
  FeatureStatisticsToProtoConfig feature_statistics_to_proto_config;
  feature_statistics_to_proto_config.set_enum_threshold(kDefaultEnumThreshold);
  return feature_statistics_to_proto_config;
}

using ::tensorflow::metadata::v0::FeatureNameStatistics;
using ::tensorflow::metadata::v0::StringDomain;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::HasSubstr;
using testing::EqualsProto;
using testing::ParseTextProtoOrDie;

struct EnumTypeIsValidTest {
  const string name;
  const std::vector<string> values;
  const FeatureNameStatistics input;
  const bool expected;
};

StringDomain GetStringDomain(const string& name,
                             const std::vector<string>& values) {
  StringDomain string_domain;
  *string_domain.mutable_name() = name;
  for (const string& str : values) {
    *string_domain.add_value() = str;
  }
  return string_domain;
}

TEST(EnumType, IsValid) {
  const std::vector<EnumTypeIsValidTest> tests = {
      {"true_false",
       {"true", "false"},
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          unique: 3
          rank_histogram: {
            buckets: {
              label: "true"
            }
            buckets: {
              label: "false"
            }}})"),
       true},
      {"A non UTF8 string occurs",
       {"true", "false"},
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          unique: 3
          invalid_utf8_count: 1
          rank_histogram: {
            buckets: {
              label: "__BYTES_VALUE__"
            }
            buckets: {
              label: "false"
            }}})"),
       false},
      {"new value with small frequency",
       {"a", "b"},
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            tot_num_values: 5
            num_missing: 3
            num_non_missing: 5
            avg_num_values: 1
            max_num_values: 2
          }
          unique: 3
          rank_histogram: {
            buckets: {
              label: "wacky"
              sample_count: 1.0e-12
            }}})"),
       false},
      {"true_false with wacky value",
       {"true", "false"},
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            num_missing: 3
            tot_num_values: 5
            num_non_missing: 5
            avg_num_values: 1
            max_num_values: 2
          }
          unique: 3
          rank_histogram: {
            buckets: {
              label: "true"
              sample_count: 4
            }
            buckets: {
              label: "wacky"
              sample_count: 1
            }}})"),
       false},
      {"false_only",
       {"false"},
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          unique: 3
          rank_histogram: {
            buckets: {
              label: "false"
            }}})"),
       true}};
  for (const auto& test : tests) {
    StringDomain string_domain = GetStringDomain(test.name, test.values);
    const StringDomain original = string_domain;
    UpdateSummary result = UpdateStringDomain(
        Schema::Updater(GetDefaultFeatureStatisticsToProtoConfig()),
        testing::DatasetForTesting(test.input).feature_stats_view(), 0,
        &string_domain);
    // If it is valid, then there should be no descriptions.
    EXPECT_EQ(test.expected, result.descriptions.empty());
    if (test.expected) {
      EXPECT_FALSE(result.clear_field);
      EXPECT_THAT(string_domain, EqualsProto(original));
    }
  }
}

TEST(EnumType, IsValidWithMassConstraint) {
  const std::vector<string> kDomain = {"a", "b"};
  const std::vector<EnumTypeIsValidTest> tests = {
      {"all_mass_in_domain", kDomain,
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            tot_num_values: 10
            num_non_missing: 5
            avg_num_values: 2
          }
          unique: 2
          rank_histogram: {
            buckets: {
              label: "a"
              sample_count: 8
            }
            buckets: {
              label: "b"
              sample_count: 2
            }}})"),
       true},
      {"min_mass_in_domain", kDomain,
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            tot_num_values: 10
            num_non_missing: 2
            avg_num_values: 5
          }
          unique: 2
          rank_histogram: {
            buckets: {
              label: "a"
              sample_count: 9
            }
            buckets: {
              label: "c"
              sample_count: 1
            }}})"),
       true},
      {"not_enough_mass_in_domain", kDomain,
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            tot_num_values: 10
            num_non_missing: 10
            avg_num_values: 1
          }
          unique: 2
          rank_histogram: {
            buckets: {
              label: "a"
              sample_count: 8
            }
            buckets: {
              label: "c"
              sample_count: 2.001
            }}})"),
       false},
  };

  for (const auto& test : tests) {
    StringDomain string_domain = GetStringDomain(test.name, test.values);
    const StringDomain original = string_domain;
    UpdateSummary result = UpdateStringDomain(
        Schema::Updater(GetDefaultFeatureStatisticsToProtoConfig()),
        testing::DatasetForTesting(test.input).feature_stats_view(), 0.2,
        &string_domain);
    EXPECT_EQ(test.expected, result.descriptions.empty());
    if (test.expected) {
      EXPECT_FALSE(result.clear_field) << "test: " << test.name;
      EXPECT_THAT(string_domain, EqualsProto(original))
          << "test: " << test.name;
    }
  }
}

struct EnumTypeUpdateTest {
  const string name;
  const std::vector<string> values;
  const FeatureNameStatistics input;
  const StringDomain expected;
  const bool expected_clear_field;
  const size_t expected_descriptions_size;
};

TEST(EnumType, Update) {
  const std::vector<EnumTypeUpdateTest> tests = {
      {"true_false, constant",
       {"true", "false"},
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          unique: 3
          rank_histogram: {
            buckets: {
              label: "true"
            }
            buckets: {
              label: "false"
            }}})"),
       ParseTextProtoOrDie<StringDomain>(
           R"( name: "true_false, constant" value: "true" value: "false")"),
       false,
       0},
      {"non_UTF8",
       {"true", "false"},
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          unique: 3
          invalid_utf8_count: 1
          rank_histogram: {
            buckets: {
              label: "__BYTES_VALUE__"
            }
            buckets: {
              label: "false"
            }}})"),
       ParseTextProtoOrDie<StringDomain>(
           R"( name: "non_UTF8" value: "true" value: "false")"),
       true,
       1},
      {"true_false_add_other",
       {"true", "false"},
       ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: STRING
        string_stats: {
          common_stats: {
            num_missing: 3
            max_num_values: 2
          }
          unique: 3
          rank_histogram: {
            buckets: {
              label: "true"
            }
            buckets: {
              label: "other"
            }
            buckets: {
              label: "false"
            }}})"),
       ParseTextProtoOrDie<StringDomain>(
           R"( name: "true_false_add_other" value: "true" value: "false"
           value: "other")"),
       false,
       1}};
  for (const auto& test : tests) {
    StringDomain to_modify = GetStringDomain(test.name, test.values);
    std::vector<Description> descriptions;
    UpdateSummary summary = UpdateStringDomain(
        Schema::Updater(GetDefaultFeatureStatisticsToProtoConfig()),
        testing::DatasetForTesting(test.input).feature_stats_view(), 0,
        &to_modify);
    EXPECT_EQ(summary.clear_field, test.expected_clear_field)
        << "Name: " << test.name;
    EXPECT_EQ(summary.descriptions.size(), test.expected_descriptions_size)
        << " Name: " << test.expected_descriptions_size;
    EXPECT_THAT(to_modify, EqualsProto(test.expected)) << "Name: " << test.name;
  }
}

TEST(EnumType, SurfaceFrequenciesOfMissingValues) {
  // Case: Percentage of new value >= 1%
  {
    StringDomain string_domain = ParseTextProtoOrDie<StringDomain>(
        R"(name: "MyEnum" value: "alpha" value: "beta")");
    std::vector<Description> descriptions;
    UpdateSummary summary =
        UpdateStringDomain(
          Schema::Updater(GetDefaultFeatureStatisticsToProtoConfig()),
          testing::DatasetForTesting(
          ParseTextProtoOrDie<FeatureNameStatistics>(R"(
            name: 'bar'
            type: STRING
            string_stats: {
              common_stats: {
                tot_num_values: 10
                num_missing: 3
                max_num_values: 2
                avg_num_values: 1
              }
              unique: 3
              rank_histogram: {
                buckets: {
                  label: "alpha"
                  sample_count: 7
                }
                buckets: {
                  label: "gamma"
                  sample_count: 3}}})"))
                               .feature_stats_view(),
                           0, &string_domain);
    EXPECT_THAT(summary.descriptions,
                ElementsAre(Field(&Description::long_description,
                                  HasSubstr("gamma (~30%)"))));
  }

  // Case: percentage of value < 1%.
  {
    StringDomain string_domain = ParseTextProtoOrDie<StringDomain>(
        R"(name: "MyEnum" value: "alpha" value: "beta")");

    UpdateSummary summary =
        UpdateStringDomain(
          Schema::Updater(GetDefaultFeatureStatisticsToProtoConfig()),
          testing::DatasetForTesting(
          ParseTextProtoOrDie<FeatureNameStatistics>(R"(
            name: 'bar'
            type: STRING
            string_stats: {
              common_stats: {
                tot_num_values: 124
                num_non_missing: 10
                num_missing: 3
                max_num_values: 2
                avg_num_values: 1
              }
              unique: 3
              rank_histogram: {
                buckets: {
                  label: "alpha"
                  sample_count: 123
                }
                buckets: {
                  label: "gamma"
                  sample_count: 0.05}}})"))
                               .feature_stats_view(),
                           0, &string_domain);
    EXPECT_THAT(summary.descriptions,
                ElementsAre(Field(&Description::long_description,
                                  HasSubstr("gamma (<1%)"))));
  }
}

TEST(EnumType, DomainSizeLimit) {
  // Case: Percentage of new value >= 1%
  {
    StringDomain string_domain = ParseTextProtoOrDie<StringDomain>(
        R"(name: "MyEnum" value: "alpha" value: "beta")");
    std::vector<Description> descriptions;
    FeatureStatisticsToProtoConfig config
        = GetDefaultFeatureStatisticsToProtoConfig();
    config.set_enum_delete_threshold(1);
    const UpdateSummary summary =
        UpdateStringDomain(
          Schema::Updater(config),
          testing::DatasetForTesting(
          ParseTextProtoOrDie<FeatureNameStatistics>(R"(
            name: 'bar'
            type: STRING
            string_stats: {
              common_stats: {
                tot_num_values: 10
                num_missing: 3
                max_num_values: 2
                avg_num_values: 1
              }
              unique: 3
              rank_histogram: {
                buckets: {
                  label: "alpha"
                  sample_count: 7
                }
                buckets: {
                  label: "gamma"
                  sample_count: 3}}})"))
                               .feature_stats_view(),
                           0, &string_domain);
    EXPECT_TRUE(summary.clear_field);
    EXPECT_THAT(summary.descriptions,
                ElementsAre(
                    Field(&Description::long_description,
                                  HasSubstr("gamma (~30%)")),
                    Field(&Description::long_description,
                                  HasSubstr("too many values"))
                    ));
  }
  // Don't delete.
  {
    StringDomain string_domain = ParseTextProtoOrDie<StringDomain>(
        R"(name: "MyEnum" value: "alpha" value: "beta")");
    std::vector<Description> descriptions;
    FeatureStatisticsToProtoConfig config
        = GetDefaultFeatureStatisticsToProtoConfig();
    config.set_enum_delete_threshold(10);
    const UpdateSummary summary =
        UpdateStringDomain(
          Schema::Updater(config),
          testing::DatasetForTesting(
          ParseTextProtoOrDie<FeatureNameStatistics>(R"(
            name: 'bar'
            type: STRING
            string_stats: {
              common_stats: {
                tot_num_values: 10
                num_missing: 3
                max_num_values: 2
                avg_num_values: 1
              }
              unique: 3
              rank_histogram: {
                buckets: {
                  label: "alpha"
                  sample_count: 7
                }
                buckets: {
                  label: "gamma"
                  sample_count: 3}}})"))
                               .feature_stats_view(),
                           0, &string_domain);
    EXPECT_FALSE(summary.clear_field);
    EXPECT_THAT(summary.descriptions,
                ElementsAre(Field(&Description::long_description,
                                  HasSubstr("gamma (~30%)"))));
  }
}

TEST(Enum, Add) {
  StringDomain to_modify = ParseTextProtoOrDie<StringDomain>("name: 'MyEnum'");
  const FeatureNameStatistics stats =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: {
             num_missing: 3
             max_num_values: 2
           }
           unique: 3
           rank_histogram: {
             buckets: {
               label: "alpha"
               sample_count: 123
             }
             buckets: {
               label: "beta"
               sample_count: 234 }}})");
  const testing::DatasetForTesting dataset_for_testing(stats);
  UpdateStringDomain(
      Schema::Updater(GetDefaultFeatureStatisticsToProtoConfig()),
      dataset_for_testing.feature_stats_view(), 0, &to_modify);
  EXPECT_THAT(to_modify, EqualsProto(R"(
      name: "MyEnum"
      value: "alpha"
      value: "beta")"));
}

TEST(Enum, GetMissingUnweighted) {
  StringDomain to_modify = ParseTextProtoOrDie<StringDomain>("name: 'MyEnum'");
  const FeatureNameStatistics stats =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: {
             num_missing: 3
             max_num_values: 2
           }
           unique: 3
           rank_histogram: {
             buckets: {
               label: "alpha"
               sample_count: 123
             }
             buckets: {
               label: "beta"
               sample_count: 234 }}})");
  const testing::DatasetForTesting dataset(stats);
  UpdateStringDomain(
      Schema::Updater(GetDefaultFeatureStatisticsToProtoConfig()),
      dataset.feature_stats_view(), 0, &to_modify);
  EXPECT_THAT(to_modify,
              EqualsProto(R"(name: "MyEnum" value: "alpha" value: "beta")"));
}

// Try get_missing with the weight_by field.
TEST(Enum, GetMissingWeighted) {
  StringDomain to_modify = ParseTextProtoOrDie<StringDomain>("name:'MyEnum'");
  const FeatureNameStatistics stats =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: {
             num_missing: 3
             max_num_values: 2
           }
           unique: 3
           weighted_string_stats: {
           rank_histogram: {
             buckets: {
               label: "alpha"
               sample_count: 123
             }
             buckets: {
               label: "beta"
               sample_count: 234 }}}})");
  const testing::DatasetForTesting dataset(stats, true);
  UpdateStringDomain(
      Schema::Updater(GetDefaultFeatureStatisticsToProtoConfig()),
      dataset.feature_stats_view(), 0, &to_modify);
  EXPECT_THAT(to_modify,
              EqualsProto(R"(name: "MyEnum" value: "alpha" value: "beta")"));
}

TEST(Enum, IsSimilar) {
  StringDomain domain = ParseTextProtoOrDie<StringDomain>(R"(
      name: "EnumA"
      value: "foo"
      value: "bar"
      value: "baz"
      )");
  EXPECT_TRUE(IsSimilarStringDomain(domain, domain, EnumsSimilarConfig()));
}

TEST(Enum, IsCandidate) {
  const FeatureNameStatistics stats =
      ParseTextProtoOrDie<FeatureNameStatistics>(R"(
         name: 'bar'
         type: STRING
         string_stats: {
           common_stats: {
             num_missing: 3
             max_num_values: 2
           }
           unique: 3
           rank_histogram: {
             buckets: {
               label: "alpha"
             }
             buckets: {
               label: "beta"}}})");
  const testing::DatasetForTesting dataset(stats);
  EXPECT_TRUE(IsStringDomainCandidate(dataset.feature_stats_view(), 2));
  EXPECT_FALSE(IsStringDomainCandidate(dataset.feature_stats_view(), 1));
}

TEST(UpdateStringDomainSelf, OneRepeatEnd) {
    StringDomain to_modify = ParseTextProtoOrDie<StringDomain>(R"(
        name:'MyEnum'
        value: "alpha"
        value: "beta"
        value: "alpha")");
    UpdateStringDomainSelf(&to_modify);
    EXPECT_THAT(to_modify,
                EqualsProto(R"(name: 'MyEnum' value: "alpha" value: "beta")"));
}

TEST(UpdateStringDomainSelf, NoProblems) {
    StringDomain to_modify = ParseTextProtoOrDie<StringDomain>(R"(
        name:'MyEnum'
        value: "alpha"
        value: "beta")");
    UpdateStringDomainSelf(&to_modify);
    EXPECT_THAT(to_modify,
                EqualsProto(R"(name: 'MyEnum' value: "alpha" value: "beta")"));
}
}  // namespace
}  // namespace data_validation
}  // namespace tensorflow
