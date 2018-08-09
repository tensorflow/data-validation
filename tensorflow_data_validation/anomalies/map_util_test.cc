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

#include "tensorflow_data_validation/anomalies/map_util.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data_validation {
namespace {

TEST(MapUtilTest, ContainsKeyMapKeyPresent) {
  const std::map<string, double> a = {{"a", 3.0}, {"b", 1.0}};
  EXPECT_TRUE(ContainsKey(a, "a"));
}

TEST(MapUtilTest, ContainsKeyMapKeyAbsent) {
  const std::map<string, double> a = {{"a", 3.0}, {"b", 1.0}};
  EXPECT_FALSE(ContainsKey(a, "c"));
}

TEST(MapUtilTest, ContainsKeyMapEmpty) {
  const std::map<string, double> a = {};
  EXPECT_FALSE(ContainsKey(a, "a"));
}

TEST(MapUtilTest, ContainsKeySetKeyPresent) {
  const std::set<string> a = {"a", "b"};
  EXPECT_TRUE(ContainsKey(a, "a"));
}

TEST(MapUtilTest, ContainsKeySetKeyAbsent) {
  const std::set<string> a = {"a", "b"};
  EXPECT_FALSE(ContainsKey(a, "c"));
}

TEST(MapUtilTest, ContainsKeySetEmpty) {
  const std::set<string> a = {};
  EXPECT_FALSE(ContainsKey(a, "a"));
}

TEST(MapUtilTest, SumValues) {
  const std::map<string, double> a = {{"a", 3.0}, {"b", 1.0}};
  double result = SumValues(a);
  EXPECT_NEAR(result, 4.0, 1.0e-7);
}

TEST(MapUtilTest, SumValuesEmpty) {
  const std::map<string, double> a = {};
  double result = SumValues(a);
  EXPECT_NEAR(result, 0.0, 1.0e-7);
}

TEST(MapUtilTest, GetValuesFromMap) {
  const std::map<string, double> a = {{"a", 3.0}, {"b", 1.0}};
  const std::vector<double> result = GetValuesFromMap(a);
  EXPECT_THAT(result, testing::ElementsAre(3.0, 1.0));
}

TEST(MapUtilTest, GetValuesFromMapEmpty) {
  const std::map<string, double> a = {};
  const std::vector<double> result = GetValuesFromMap(a);
  EXPECT_THAT(result, testing::ElementsAre());
}

TEST(MapUtilTest, GetKeysFromMap) {
  const std::map<string, double> a = {{"a", 3.0}, {"b", 1.0}};
  const std::vector<string> result = GetKeysFromMap(a);
  EXPECT_THAT(result, testing::ElementsAre("a", "b"));
}

TEST(MapUtilTest, GetKeysFromMapEmpty) {
  const std::map<string, double> a = {};
  const std::vector<string> result = GetKeysFromMap(a);
  EXPECT_THAT(result, testing::ElementsAre());
}

TEST(MapUtilTest, Normalize) {
  const std::map<string, double> a = {{"a", 3.0}, {"b", 1.0}};
  const std::map<string, double> result = Normalize(a);
  EXPECT_THAT(result,
              testing::ElementsAre(std::pair<string, double>("a", 0.75),
                                   std::pair<string, double>("b", 0.25)));
}

TEST(MapUtilTest, NormalizeAllZeros) {
  const std::map<string, double> a = {{"a", 0.0}, {"b", 0.0}};
  const std::map<string, double> result = Normalize(a);
  EXPECT_THAT(result,
              testing::ElementsAre(std::pair<string, double>("a", 0.0),
                                   std::pair<string, double>("b", 0.0)));
}

TEST(MapUtilTest, NormalizeEmpty) {
  const std::map<string, double> a = {};
  const std::map<string, double> result = Normalize(a);
  EXPECT_THAT(result, testing::ElementsAre());
}

TEST(MapUtilTest, GetDifference) {
  const std::map<string, double> a = {{"a", 3.0}, {"b", 1.0}};
  const std::map<string, double> b = {{"c", 1.0}, {"b", 1.0}};
  const std::map<string, double> c = GetDifference(a, b);
  EXPECT_THAT(c, testing::ElementsAre(std::pair<string, double>("a", 3.0),
                                      std::pair<string, double>("b", 0.0),
                                      std::pair<string, double>("c", -1.0)));
}

TEST(MapUtilTest, GetDifferenceEmpty) {
  const std::map<string, double> a = {};
  const std::map<string, double> b = {};
  const std::map<string, double> c = GetDifference(a, b);
  EXPECT_THAT(c, testing::ElementsAre());
}

TEST(MapUtilTest, IncrementMap) {
  const std::map<string, double> a = {{"c", 1.0}, {"b", 1.0}};
  std::map<string, double> b = {{"a", 3.0}, {"b", 1.0}};
  IncrementMap(a, &b);
  EXPECT_THAT(b, testing::ElementsAre(std::pair<string, double>("a", 3.0),
                                      std::pair<string, double>("b", 2.0),
                                      std::pair<string, double>("c", 1.0)));
}

TEST(MapUtilTest, IncrementMapEmpty) {
  const std::map<string, double> a;
  std::map<string, double> b;
  IncrementMap(a, &b);
  EXPECT_THAT(b, testing::ElementsAre());
}

TEST(MapUtilTest, GetSum) {
  const std::map<string, double> a = {{"a", 3.0}, {"b", 1.0}};
  const std::map<string, double> b = {{"c", 1.0}, {"b", 1.0}};
  const std::map<string, double> c = GetSum(a, b);
  EXPECT_THAT(c, testing::ElementsAre(std::pair<string, double>("a", 3.0),
                                      std::pair<string, double>("b", 2.0),
                                      std::pair<string, double>("c", 1.0)));
}

TEST(MapUtilTest, GetSumEmpty) {
  const std::map<string, double> a = {};
  const std::map<string, double> b = {};
  const std::map<string, double> c = GetSum(a, b);
  EXPECT_THAT(c, testing::ElementsAre());
}

TEST(MapUtilTest, MapValues) {
  const std::map<string, double> a = {{"a", 3.0}, {"b", 1.0}};
  const std::map<string, double> c =
      MapValues(a, [](double a) { return a + 1.0; });
  EXPECT_THAT(c, testing::ElementsAre(std::pair<string, double>("a", 4.0),
                                      std::pair<string, double>("b", 2.0)));
}

TEST(MapUtilTest, MapValuesEmpty) {
  const std::map<string, double> a = {};
  const std::map<string, double> c =
      MapValues(a, [](double a) { return a + 1.0; });
  EXPECT_THAT(c, testing::ElementsAre());
}

}  // namespace
}  // namespace data_validation
}  // namespace tensorflow
