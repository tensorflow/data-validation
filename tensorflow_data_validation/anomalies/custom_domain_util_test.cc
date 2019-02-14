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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow_data_validation/anomalies/test_util.h"

namespace tensorflow {
namespace data_validation {

namespace {

using ::tensorflow::metadata::v0::CustomStatistic;
using ::tensorflow::metadata::v0::Feature;
using ::testing::Test;

CustomStatistic DomainInfoStatistic(const string& value) {
  CustomStatistic custom_stat;
  custom_stat.set_name("domain_info");
  custom_stat.set_str(value);
  return custom_stat;
}

TEST(CustomDomainUtilTest, FailureOnNoDomainInfoCustomStat) {
  Feature feature;
  CustomStatistic custom_stat;
  custom_stat.set_name("some_other_name");
  custom_stat.set_str("natural_language_domain");
  EXPECT_FALSE(BestEffortUpdateCustomDomain(
      std::vector<CustomStatistic>({custom_stat}), &feature));
  EXPECT_EQ(feature.domain_info_case(),
            tensorflow::metadata::v0::Feature::DOMAIN_INFO_NOT_SET);
}

TEST(CustomDomainUtilTest, SuccessOnEmptyFeature) {
  Feature feature;
  EXPECT_TRUE(BestEffortUpdateCustomDomain(
      std::vector<CustomStatistic>(
          {DomainInfoStatistic("natural_language_domain {}")}),
      &feature));
  EXPECT_TRUE(feature.has_natural_language_domain());
}

TEST(CustomDomainUtilTest, FailureOnFeatureWithDomain) {
  Feature feature;
  feature.mutable_string_domain();
  EXPECT_FALSE(BestEffortUpdateCustomDomain(
      std::vector<CustomStatistic>(
          {DomainInfoStatistic("natural_language_domain {}")}),
      &feature));
  EXPECT_TRUE(feature.has_string_domain());
}

TEST(CustomDomainUtilTest, FailureOnMultipleDomainInfosFeature) {
  Feature feature;
  EXPECT_FALSE(BestEffortUpdateCustomDomain(
      std::vector<CustomStatistic>(
          {DomainInfoStatistic("natural_language_domain {}"),
           DomainInfoStatistic("natural_language_domain {}")}),
      &feature));
  EXPECT_EQ(feature.domain_info_case(),
            tensorflow::metadata::v0::Feature::DOMAIN_INFO_NOT_SET);
}

TEST(CustomDomainUtilTest, FailureOnInvalidDomainValue) {
  Feature feature;

  EXPECT_FALSE(BestEffortUpdateCustomDomain(
      std::vector<CustomStatistic>({DomainInfoStatistic("")}), &feature));
  EXPECT_EQ(feature.domain_info_case(),
            tensorflow::metadata::v0::Feature::DOMAIN_INFO_NOT_SET);

  EXPECT_FALSE(BestEffortUpdateCustomDomain(
      std::vector<CustomStatistic>({DomainInfoStatistic("This is not valid")}),
      &feature));
  EXPECT_EQ(feature.domain_info_case(),
            tensorflow::metadata::v0::Feature::DOMAIN_INFO_NOT_SET);

  EXPECT_FALSE(BestEffortUpdateCustomDomain(
      std::vector<CustomStatistic>({DomainInfoStatistic(
          "name: 'It should not set other fields!' image_domain {} ")}),
      &feature));
  EXPECT_EQ(feature.domain_info_case(),
            tensorflow::metadata::v0::Feature::DOMAIN_INFO_NOT_SET);
}

}  // namespace

}  // namespace data_validation
}  // namespace tensorflow
