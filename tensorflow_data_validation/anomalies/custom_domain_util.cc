/* Copyright 2019 Google LLC

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

#include "tensorflow_data_validation/anomalies/custom_domain_util.h"

#include <string>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace data_validation {
namespace {

// LINT.IfChange
constexpr char kDomainInfo[] = "domain_info";
// LINT.ThenChange(../utils/stats_util.py)

bool ParseCustomDomainInfo(const string& domain_info,
                           tensorflow::metadata::v0::Feature* feature) {
  // Temporary feature for parsing domain_info.
  tensorflow::metadata::v0::Feature domain_info_feature;
  if (!::tensorflow::protobuf::TextFormat::ParseFromString(
          domain_info, &domain_info_feature)) {
    return false;
  }
  // Ensure only one field is set
  std::vector<const ::tensorflow::protobuf::FieldDescriptor*> fields_set;
  feature->GetReflection()->ListFields(domain_info_feature, &fields_set);
  // Ensure only one field is set, which is part of the domain_info oneof.
  if (fields_set.size() != 1 || fields_set[0]->containing_oneof() == nullptr ||
      fields_set[0]->containing_oneof()->name() != kDomainInfo) {
    return false;
  } else {
    feature->MergeFrom(domain_info_feature);
    return true;
  }
}

}  // namespace

bool BestEffortUpdateCustomDomain(
    const std::vector<tensorflow::metadata::v0::CustomStatistic>& custom_stats,
    tensorflow::metadata::v0::Feature* feature) {
  string domain_info;
  for (const auto& custom_stat : custom_stats) {
    if (custom_stat.name() == kDomainInfo) {
      if (!domain_info.empty()) {
        LOG(ERROR) << "Duplicate 'domain_info' custom_stat [" << domain_info
                   << ", " << custom_stat.str() << "], this is a stats bug.";
        return false;
      } else {
        domain_info = custom_stat.str();
      }
    }
  }
  if (domain_info.empty()) {
    return false;
  }
  // Never override existing domain_infos with a custom domain_info for safety.
  if (feature->domain_info_case() !=
      tensorflow::metadata::v0::Feature::DOMAIN_INFO_NOT_SET) {
    LOG(INFO) << "Valid custom domain_info: " << domain_info
              << " ignored due to existing domain, for feature :"
              << feature->DebugString();
    return false;
  }
  if (!ParseCustomDomainInfo(domain_info, feature)) {
    LOG(ERROR) << "Could not parse 'domain_info' custom_stat: " << domain_info
               << ". It is expected to contain exactly one field of the "
               << "Feature.domain_info oneof, e.g: 'mid_domain {}'.";
    return false;
  }
  return true;
}

}  // namespace data_validation
}  // namespace tensorflow
