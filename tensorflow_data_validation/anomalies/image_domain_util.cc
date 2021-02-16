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

#include <set>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow_data_validation/anomalies/map_util.h"
#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tensorflow {
namespace data_validation {
namespace {

using ::tensorflow::metadata::v0::CustomStatistic;
using ::tensorflow::metadata::v0::Feature;
using ::tensorflow::metadata::v0::ImageDomain;

}  // namespace

std::vector<Description> UpdateImageDomain(
    const FeatureStatsView& feature_stats, Feature* feature) {
  std::vector<Description> results;
  const ImageDomain& image_domain = feature->image_domain();
  if (image_domain.has_minimum_supported_image_fraction()) {
    const CustomStatistic* image_format_histogram =
        feature_stats.GetCustomStatByName("image_format_histogram");
    if (image_format_histogram) {
      float supported_image_count = 0;
      float unsupported_image_count = 0;
      for (const metadata::v0::RankHistogram::Bucket& bucket :
           image_format_histogram->rank_histogram().buckets()) {
        if (bucket.label() == "UNKNOWN") {
          unsupported_image_count += bucket.sample_count();
        } else {
          supported_image_count += bucket.sample_count();
        }
      }
      float supported_image_fraction =
          supported_image_count /
          (supported_image_count + unsupported_image_count);
      const float original_minimum_supported_image_fraction =
          image_domain.minimum_supported_image_fraction();
      if (supported_image_fraction <
          original_minimum_supported_image_fraction) {
        feature->mutable_image_domain()->set_minimum_supported_image_fraction(
            supported_image_fraction);
        results.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::
                 LOW_SUPPORTED_IMAGE_FRACTION,
             "Low supported image fraction",
             absl::StrCat(
                 "Fraction of values containing TensorFlow supported "
                 "images: ",
                 std::to_string(supported_image_fraction),
                 " is lower than the threshold set in the Schema: ",
                 std::to_string(original_minimum_supported_image_fraction),
                 ".")});
      }
    } else {
      LOG(WARNING)
          << "image_domain.minimum_supported_image_fraction is specified "
             "for feature "
          << feature->name()
          << ", but there is no "
             "image_format_histogram in the statistics. You must enable "
             "semantic "
             "domain stats for the image_format_histogram to be generated.";
    }
  }
  if (image_domain.has_max_image_byte_size()) {
    const int64_t max_bytes_stat =
        feature_stats.bytes_stats().max_num_bytes_int();
    const int64_t max_allowed_bytes = image_domain.max_image_byte_size();
    if (max_bytes_stat > max_allowed_bytes) {
      feature->mutable_image_domain()->set_max_image_byte_size(max_bytes_stat);
      results.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::MAX_IMAGE_BYTE_SIZE_EXCEEDED,
           "Num bytes exceeds the max byte size.",
           absl::StrCat("The largest image has bytes: ", max_bytes_stat,
                        ". The max allowed byte size is: ", max_allowed_bytes,
                        ".")});
    }
  }
  return results;
}

}  // namespace data_validation
}  // namespace tensorflow
