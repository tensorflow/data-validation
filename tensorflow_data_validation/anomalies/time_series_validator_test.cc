/* Copyright 2023 Google LLC

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

#include "tensorflow_data_validation/anomalies/time_series_validator.h"

#include <string>

#include "testing/base/public/googletest.h"
#include <gtest/gtest.h>
#include "tensorflow_data_validation/anomalies/test_util.h"
#include "tensorflow_data_validation/google/protos/time_series_metrics.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

using tensorflow::data_validation::SliceComparisonConfig;

SliceComparisonConfig slice_config = SliceComparisonConfig::Corresponding();

namespace tensorflow {

namespace data_validation {

namespace {

using ::testing::EqualsProto;
using testing::ParseTextProtoOrDie;
using ::testing::TestWithParam;
using ::testing::UnorderedPointwise;
using ::testing::status::IsOkAndHolds;

struct TestCase {
  std::string test_name;
  metadata::v0::DatasetFeatureStatisticsList test_statistics;
  metadata::v0::DatasetFeatureStatisticsList test_reference_statistics;
  std::vector<ValidationMetrics> expected_validation_metrics;
};

using TimeseriesValidationTest = TestWithParam<TestCase>;

TEST_P(TimeseriesValidationTest, TestFeatureStatsTimeSeriesValidation) {
  const TestCase& test_case = GetParam();

  EXPECT_THAT(ValidateTimeSeriesStatistics(test_case.test_statistics,
                                           test_case.test_reference_statistics,
                                           slice_config),
              IsOkAndHolds(UnorderedPointwise(
                  EqualsProto(), test_case.expected_validation_metrics)));
}

INSTANTIATE_TEST_SUITE_P(
    TimeseriesValidationTests, TimeseriesValidationTest,
    ::testing::ValuesIn<TestCase>({
        {"SingleNumFeatureSingleSlice",
         ParseTextProtoOrDie<metadata::v0::DatasetFeatureStatisticsList>(
             R"pb(
               datasets {
                 name: "All Examples"
                 num_examples: 15
                 features {
                   path { step: 'test_feature1' }
                   type: INT
                   num_stats {
                     num_zeros: 1
                     max: 25
                     common_stats {
                       num_non_missing: 13
                       num_values_histogram {
                         buckets { high_value: 0 low_value: 0 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 25 low_value: 1 sample_count: 1 }
                         type: QUANTILES
                       }
                     }
                   }
                 }
               }
             )pb"),
         ParseTextProtoOrDie<metadata::v0::DatasetFeatureStatisticsList>(
             R"pb(
               datasets {
                 name: "All Examples"
                 num_examples: 18
                 features {
                   path { step: 'test_feature1' }
                   type: INT
                   num_stats {
                     num_zeros: 2
                     max: 20
                     common_stats {
                       num_non_missing: 15
                       num_values_histogram {
                         buckets { high_value: 0 low_value: 0 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 3 }
                         buckets { high_value: 1 low_value: 1 sample_count: 3 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 20 low_value: 1 sample_count: 1 }
                         type: QUANTILES
                       }
                     }
                   }
                 }
               })pb"),
         {ParseTextProtoOrDie<ValidationMetrics>(
             R"pb(feature_metric {
                    feature_name { name: 'test_feature1' }
                    metric { metric_name: 'num_examples' value: 15 }
                    metric { metric_name: "num_not_missing" value: 13 }
                    metric {
                      metric_name: "num_values_jensen_shannon_divergence"
                      value: 0.01409979433847891
                    }
                  }
                  source { slice { slice_name: 'All Examples' } }
                  reference_source { slice { slice_name: 'All Examples' } }
             )pb")}},
        {"MultiTypeFeatureSingleSlice",
         ParseTextProtoOrDie<metadata::v0::DatasetFeatureStatisticsList>(
             R"pb(
               datasets {
                 name: "All Examples"
                 num_examples: 15
                 features {
                   path { step: 'test_feature1' }
                   type: INT
                   num_stats {
                     num_zeros: 1
                     max: 25
                     common_stats {
                       num_non_missing: 13
                       num_values_histogram {
                         buckets { high_value: 0 low_value: 0 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 25 low_value: 1 sample_count: 1 }
                         type: QUANTILES
                       }
                     }
                   }
                 }
                 features {
                   path { step: 'test_feature2' }
                   type: STRING
                   string_stats {
                     unique: 1
                     common_stats {
                       num_non_missing: 14
                       num_values_histogram {
                         buckets { high_value: 0 low_value: 0 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 25 low_value: 1 sample_count: 1 }
                         type: QUANTILES
                       }
                     }
                   }
                 }
               }
             )pb"),
         ParseTextProtoOrDie<metadata::v0::DatasetFeatureStatisticsList>(
             R"pb(
               datasets {
                 name: "All Examples"
                 num_examples: 18
                 features {
                   path { step: 'test_feature1' }
                   type: INT
                   num_stats {
                     num_zeros: 2
                     max: 20
                     common_stats {
                       num_non_missing: 15
                       num_values_histogram {
                         buckets { high_value: 0 low_value: 0 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 3 }
                         buckets { high_value: 1 low_value: 1 sample_count: 3 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 20 low_value: 1 sample_count: 1 }
                         type: QUANTILES
                       }
                     }
                   }
                 }
                 features {
                   path { step: 'test_feature2' }
                   type: STRING
                   string_stats {
                     unique: 2
                     common_stats {
                       num_non_missing: 15
                       num_values_histogram {
                         buckets { high_value: 0 low_value: 0 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 3 }
                         buckets { high_value: 1 low_value: 1 sample_count: 3 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 20 low_value: 1 sample_count: 1 }
                         type: QUANTILES
                       }
                     }
                   }
                 }
               })pb"),
         {ParseTextProtoOrDie<ValidationMetrics>(
             R"pb(feature_metric {
                    feature_name { name: 'test_feature1' }
                    metric { metric_name: 'num_examples' value: 15 }
                    metric { metric_name: "num_not_missing" value: 13 }
                    metric {
                      metric_name: "num_values_jensen_shannon_divergence"
                      value: 0.01409979433847891
                    }
                  }
                  feature_metric {
                    feature_name { name: 'test_feature2' }
                    metric { metric_name: 'num_examples' value: 15 }
                    metric { metric_name: "num_not_missing" value: 14 }
                    metric {
                      metric_name: "num_values_jensen_shannon_divergence"
                      value: 0.015308307959065924
                    }
                  }
                  source { slice { slice_name: 'All Examples' } }
                  reference_source { slice { slice_name: 'All Examples' } }
             )pb")}},
        {"SingleFeatureMultiSlice",
         ParseTextProtoOrDie<metadata::v0::DatasetFeatureStatisticsList>(
             R"pb(
               datasets {
                 name: "All Examples"
                 num_examples: 15
                 features {
                   path { step: 'test_feature1' }
                   type: INT
                   num_stats {
                     num_zeros: 1
                     max: 25
                     common_stats {
                       num_non_missing: 13
                       num_values_histogram {
                         buckets { high_value: 0 low_value: 0 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 25 low_value: 1 sample_count: 1 }
                         type: QUANTILES
                       }
                     }
                   }
                 }
               }
               datasets {
                 name: "Slice1"
                 num_examples: 16
                 features {
                   path { step: 'test_feature1' }
                   type: INT
                   num_stats {
                     num_zeros: 1
                     max: 25
                     common_stats {
                       num_non_missing: 14
                       num_values_histogram {
                         buckets { high_value: 0 low_value: 0 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 25 low_value: 1 sample_count: 1 }
                         type: QUANTILES
                       }
                     }
                   }
                 }
               }
             )pb"),
         ParseTextProtoOrDie<metadata::v0::DatasetFeatureStatisticsList>(
             R"pb(
               datasets {
                 name: "All Examples"
                 num_examples: 18
                 features {
                   path { step: 'test_feature1' }
                   type: INT
                   num_stats {
                     num_zeros: 2
                     max: 20
                     common_stats {
                       num_non_missing: 15
                       num_values_histogram {
                         buckets { high_value: 0 low_value: 0 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 3 }
                         buckets { high_value: 1 low_value: 1 sample_count: 3 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 20 low_value: 1 sample_count: 1 }
                         type: QUANTILES
                       }
                     }
                   }
                 }
               }
               datasets {
                 name: "Slice1"
                 num_examples: 10
                 features {
                   path { step: 'test_feature1' }
                   type: INT
                   num_stats {
                     num_zeros: 1
                     max: 20
                     common_stats {
                       num_non_missing: 9
                       num_values_histogram {
                         buckets { high_value: 0 low_value: 0 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 0 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 0 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 20 low_value: 1 sample_count: 1 }
                         type: QUANTILES
                       }
                     }
                   }
                 }
               })pb"),
         {ParseTextProtoOrDie<ValidationMetrics>(
              R"pb(feature_metric {
                     feature_name { name: 'test_feature1' }
                     metric { metric_name: 'num_examples' value: 15 }
                     metric { metric_name: "num_not_missing" value: 13 }
                     metric {
                       metric_name: "num_values_jensen_shannon_divergence"
                       value: 0.01409979433847891
                     }
                   }
                   source { slice { slice_name: 'All Examples' } }
                   reference_source { slice { slice_name: 'All Examples' } }
              )pb"),
          ParseTextProtoOrDie<ValidationMetrics>(
              R"pb(feature_metric {
                     feature_name { name: 'test_feature1' }
                     metric { metric_name: 'num_examples' value: 16 }
                     metric { metric_name: "num_not_missing" value: 14 }
                     metric {
                       metric_name: "num_values_jensen_shannon_divergence"
                       value: 0.018489712134710696
                     }
                   }
                   source { slice { slice_name: 'Slice1' } }
                   reference_source { slice { slice_name: 'Slice1' } }
              )pb")}},
        {"NoMatchingSliceInReferenceStats",
         ParseTextProtoOrDie<metadata::v0::DatasetFeatureStatisticsList>(
             R"pb(
               datasets {
                 name: "All Examples"
                 num_examples: 15
                 features {
                   path { step: 'test_feature1' }
                   type: INT
                   num_stats {
                     num_zeros: 1
                     max: 25
                     common_stats {
                       num_non_missing: 13
                       num_values_histogram {
                         buckets { high_value: 0 low_value: 0 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 25 low_value: 1 sample_count: 1 }
                         type: QUANTILES
                       }
                     }
                   }
                 }
               }
             )pb"),
         ParseTextProtoOrDie<metadata::v0::DatasetFeatureStatisticsList>(
             R"pb(
               datasets {
                 name: "Slice1"
                 num_examples: 18
                 features {
                   path { step: 'test_feature1' }
                   type: INT
                   num_stats {
                     num_zeros: 2
                     max: 20
                     common_stats {
                       num_non_missing: 15
                       num_values_histogram {
                         buckets { high_value: 0 low_value: 0 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 3 }
                         buckets { high_value: 1 low_value: 1 sample_count: 3 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 20 low_value: 1 sample_count: 1 }
                         type: QUANTILES
                       }
                     }
                   }
                 }
               })pb"),
         {ParseTextProtoOrDie<ValidationMetrics>(
             R"pb(feature_metric {
                    feature_name { name: 'test_feature1' }
                    metric { metric_name: 'num_examples' value: 15 }
                    metric { metric_name: "num_not_missing" value: 13 }
                  }
                  source { slice { slice_name: 'All Examples' } }
                  reference_source { slice { slice_name: 'All Examples' } }
             )pb")}},
        {"NoMatchingFeatureInReferenceStats",
         ParseTextProtoOrDie<metadata::v0::DatasetFeatureStatisticsList>(
             R"pb(
               datasets {
                 name: "All Examples"
                 num_examples: 15
                 features {
                   path { step: 'test_feature1' }
                   type: INT
                   num_stats {
                     num_zeros: 1
                     max: 25
                     common_stats {
                       num_non_missing: 13
                       num_values_histogram {
                         buckets { high_value: 0 low_value: 0 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 25 low_value: 1 sample_count: 1 }
                         type: QUANTILES
                       }
                     }
                   }
                 }
               }
             )pb"),
         ParseTextProtoOrDie<metadata::v0::DatasetFeatureStatisticsList>(
             R"pb(
               datasets {
                 name: "All Examples"
                 num_examples: 18
                 features {
                   path { step: 'test_feature2' }
                   type: INT
                   num_stats {
                     num_zeros: 2
                     max: 20
                     common_stats {
                       num_non_missing: 15
                       num_values_histogram {
                         buckets { high_value: 0 low_value: 0 sample_count: 2 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 3 }
                         buckets { high_value: 1 low_value: 1 sample_count: 3 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 1 low_value: 1 sample_count: 1 }
                         buckets { high_value: 20 low_value: 1 sample_count: 1 }
                         type: QUANTILES
                       }
                     }
                   }
                 }
               })pb"),
         {ParseTextProtoOrDie<ValidationMetrics>(
             R"pb(feature_metric {
                    feature_name { name: 'test_feature1' }
                    metric { metric_name: 'num_examples' value: 15 }
                    metric { metric_name: "num_not_missing" value: 13 }
                  }
                  source { slice { slice_name: 'All Examples' } }
                  reference_source { slice { slice_name: 'All Examples' } }
             )pb")}},
    }),
    [](const ::testing::TestParamInfo<TimeseriesValidationTest::ParamType>&
           info) { return info.param.test_name; });

}  // namespace

}  // namespace data_validation

}  // namespace tensorflow
