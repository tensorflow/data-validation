/* Copyright 2020 Google LLC

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

#include "tensorflow_data_validation/anomalies/image_domain_util.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow_data_validation/anomalies/statistics_view_test_util.h"
#include "tensorflow_data_validation/anomalies/test_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {
namespace {

using ::tensorflow::metadata::v0::ImageDomain;
using ::tensorflow::metadata::v0::Feature;
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using testing::EqualsProto;
using testing::ParseTextProtoOrDie;

// Used for friend for ImageType.
class ImageTypeTest : public ::testing::Test {};

struct ImageTypeIsValidTest {
  const string name;
  const ImageDomain original;
  const FeatureNameStatistics input;
};

TEST_F(ImageTypeTest, IsValidWithNoSupportedImageFraction) {
  Feature feature;
  *feature.mutable_image_domain() = ParseTextProtoOrDie<ImageDomain>("");
  const Feature original_feature = feature;

  std::vector<Description> description = UpdateImageDomain(
      testing::DatasetForTesting(ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: BYTES
        bytes_stats: {
          common_stats: {
            num_non_missing: 10
            min_num_values: 1
            max_num_values: 1
          }}
        custom_stats: {
          name: 'image_format_histogram'
          rank_histogram: {
            buckets: {
              label: 'jpeg'
              sample_count: 5
            }
            buckets: {
              label: 'png'
              sample_count: 3
            }
            buckets: {
              label: 'UNKNOWN'
              sample_count: 2
            }
          }
        })")).feature_stats_view(), &feature);

  // IsValid is the equivalent of an empty description.
  EXPECT_EQ(description.empty(), true);
  // If it is valid, then the schema shouldn't be updated.
  EXPECT_THAT(feature, EqualsProto(original_feature));
  }

TEST_F(ImageTypeTest, IsValidWithSupportedImageFraction) {
  Feature feature;
  *feature.mutable_image_domain() = ParseTextProtoOrDie<ImageDomain>(
           "minimum_supported_image_fraction: 0.85");
  const Feature original_feature = feature;

  std::vector<Description> description = UpdateImageDomain(
      testing::DatasetForTesting(ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: BYTES
        bytes_stats: {
          common_stats: {
            num_non_missing: 10
            min_num_values: 1
            max_num_values: 1
          }}
        custom_stats: {
          name: 'image_format_histogram'
          rank_histogram: {
            buckets: {
              label: 'jpeg'
              sample_count: 6
            }
            buckets: {
              label: 'png'
              sample_count: 3
            }
            buckets: {
              label: 'UNKNOWN'
              sample_count: 1
            }
          }
        })")).feature_stats_view(), &feature);

  // IsValid is the equivalent of an empty description.
  EXPECT_EQ(description.empty(), true);
  // If it is valid, then the schema shouldn't be updated.
  EXPECT_THAT(feature, EqualsProto(original_feature));
  }
TEST_F(ImageTypeTest, IsInvalidWithSupportedImageFraction) {
  Feature feature;
  *feature.mutable_image_domain() = ParseTextProtoOrDie<ImageDomain>(
           "minimum_supported_image_fraction: 0.85");
  const Feature original_feature = feature;

  std::vector<Description> description = UpdateImageDomain(
      testing::DatasetForTesting(ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: BYTES
        bytes_stats: {
          common_stats: {
            num_non_missing: 10
            min_num_values: 1
            max_num_values: 1
          }}
        custom_stats: {
          name: 'image_format_histogram'
          rank_histogram: {
            buckets: {
              label: 'jpeg'
              sample_count: 5
            }
            buckets: {
              label: 'png'
              sample_count: 3
            }
            buckets: {
              label: 'UNKNOWN'
              sample_count: 2
            }
          }
        })")).feature_stats_view(), &feature);

  // IsValid is the equivalent of an empty description.
  EXPECT_EQ(description.empty(), false);

  // Tests that image_domain is updated with the fraction in the histogram.
  EXPECT_NEAR(feature.image_domain().minimum_supported_image_fraction(),
              0.8, 0.001);
}
}  // namespace
}  // namespace data_validation
}  // namespace tensorflow
