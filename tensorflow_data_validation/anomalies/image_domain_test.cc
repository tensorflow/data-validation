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
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

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

  std::vector<Description> descriptions = UpdateImageDomain(
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

  // IsValid is the equivalent of an empty descriptions.
  EXPECT_EQ(descriptions.empty(), true);
  // If it is valid, then the schema shouldn't be updated.
  EXPECT_THAT(feature, EqualsProto(original_feature));
  }

TEST_F(ImageTypeTest, IsValidWithSupportedImageFraction) {
  Feature feature;
  *feature.mutable_image_domain() = ParseTextProtoOrDie<ImageDomain>(
           "minimum_supported_image_fraction: 0.85");
  const Feature original_feature = feature;

  std::vector<Description> descriptions = UpdateImageDomain(
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

  // IsValid is the equivalent of an empty descriptions.
  EXPECT_EQ(descriptions.empty(), true);
  // If it is valid, then the schema shouldn't be updated.
  EXPECT_THAT(feature, EqualsProto(original_feature));
  }

TEST_F(ImageTypeTest, IsInvalidWithSupportedImageFraction) {
  Feature feature;
  *feature.mutable_image_domain() = ParseTextProtoOrDie<ImageDomain>(
           "minimum_supported_image_fraction: 0.85");
  const Feature original_feature = feature;

  std::vector<Description> descriptions = UpdateImageDomain(
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

  Description expected = {
      tensorflow::metadata::v0::AnomalyInfo::LOW_SUPPORTED_IMAGE_FRACTION,
      "Low supported image fraction",
      "Fraction of values containing TensorFlow supported images: 0.800000 is "
      "lower than the threshold set in the Schema: 0.850000."};

  // Anomaly found.
  EXPECT_THAT(descriptions, ElementsAre(expected));

  // Tests that image_domain is updated with the fraction in the histogram.
  EXPECT_NEAR(feature.image_domain().minimum_supported_image_fraction(),
              0.8, 0.001);
}

TEST_F(ImageTypeTest, IsValidWithMaxImageByteSize) {
  Feature feature;
  *feature.mutable_image_domain() = ParseTextProtoOrDie<ImageDomain>(
           "max_image_byte_size: 1000");
  const Feature original_feature = feature;

  std::vector<Description> descriptions = UpdateImageDomain(
      testing::DatasetForTesting(ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: BYTES
        bytes_stats: {
          common_stats: {
            num_non_missing: 10
            min_num_values: 1
            max_num_values: 1
          }
          max_num_bytes_int: 10
        })")).feature_stats_view(), &feature);

  // IsValid is the equivalent of an empty descriptions.
  EXPECT_EQ(descriptions.empty(), true);
  // If it is valid, then the schema shouldn't be updated.
  EXPECT_THAT(feature, EqualsProto(original_feature));
}

TEST_F(ImageTypeTest, IsInvalidWithMaxImageByteSize) {
  Feature feature;
  *feature.mutable_image_domain() = ParseTextProtoOrDie<ImageDomain>(
           "max_image_byte_size: 1");
  const Feature original_feature = feature;

  std::vector<Description> descriptions = UpdateImageDomain(
      testing::DatasetForTesting(ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: BYTES
        bytes_stats: {
          common_stats: {
            num_non_missing: 10
            min_num_values: 1
            max_num_values: 1
          }
          max_num_bytes_int: 10
        })")).feature_stats_view(), &feature);

  Description expected =
      {tensorflow::metadata::v0::AnomalyInfo::MAX_IMAGE_BYTE_SIZE_EXCEEDED,
       "Num bytes exceeds the max byte size.",
       "The largest image has bytes: 10. The max allowed byte size is: 1."};

  // Anomaly found.
  EXPECT_THAT(descriptions, ElementsAre(expected));

  // Tests that image_domain is updated with the max byte size.
  EXPECT_EQ(feature.image_domain().max_image_byte_size(), 10);
}


TEST_F(ImageTypeTest, MultipleAnomalies) {
  Feature feature;
  *feature.mutable_image_domain() = ParseTextProtoOrDie<ImageDomain>(
           "max_image_byte_size: 1 minimum_supported_image_fraction: 0.85");
  const Feature original_feature = feature;

  std::vector<Description> descriptions = UpdateImageDomain(
      testing::DatasetForTesting(ParseTextProtoOrDie<FeatureNameStatistics>(R"(
        name: 'bar'
        type: BYTES
        bytes_stats: {
          common_stats: {
            num_non_missing: 10
            min_num_values: 1
            max_num_values: 1
          }
          max_num_bytes_int: 10
        }
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

  std::vector<Description> expected = {
      {tensorflow::metadata::v0::AnomalyInfo::LOW_SUPPORTED_IMAGE_FRACTION,
       "Low supported image fraction",
       "Fraction of values containing TensorFlow supported images: "
       "0.800000 "
       "is "
       "lower than the threshold set in the Schema: 0.850000."},
      {tensorflow::metadata::v0::AnomalyInfo::MAX_IMAGE_BYTE_SIZE_EXCEEDED,
       "Num bytes exceeds the max byte size.",
       "The largest image has bytes: 10. The max allowed "
       "byte size is: 1."}};

  // Anomalies found.
  EXPECT_THAT(descriptions, ElementsAreArray(expected));

  // Tests that image_domain is updated with their respective anomaly fields.
  EXPECT_NEAR(feature.image_domain().minimum_supported_image_fraction(),
              0.8, 0.001);
  EXPECT_EQ(feature.image_domain().max_image_byte_size(), 10);
}

}  // namespace
}  // namespace data_validation
}  // namespace tensorflow
