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
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data_validation {

double SumValues(const std::map<string, double>& input) {
  std::vector<double> values = GetValuesFromMap(input);
  return std::accumulate(values.begin(), values.end(), 0.0);
}

std::vector<double> GetValuesFromMap(const std::map<string, double>& input) {
  std::vector<double> values;
  values.reserve(input.size());
  for (const auto& pair : input) {
    values.push_back(pair.second);
  }
  return values;
}

std::vector<string> GetKeysFromMap(const std::map<string, double>& input) {
  std::vector<string> keys;
  keys.reserve(input.size());
  for (const auto& pair : input) {
    keys.push_back(pair.first);
  }
  return keys;
}

std::map<string, double> Normalize(const std::map<string, double>& input) {
  double sum = SumValues(input);
  if (sum == 0.0) {
    return input;
  }
  std::map<string, double> result;
  for (const auto& pair : input) {
    const string& key = pair.first;
    const double value = pair.second;
    result[key] = value / sum;
  }
  return result;
}

std::map<string, double> GetDifference(const std::map<string, double>& a,
                                       const std::map<string, double>& b) {
  std::map<string, double> result = a;
  for (const auto& pair_b : b) {
    const string& key_b = pair_b.first;
    const double value_b = pair_b.second;
    // If the key is not present, this will initialize it to zero.
    result[key_b] -= value_b;
  }
  return result;
}

void IncrementMap(const std::map<string, double>& a,
                  std::map<string, double>* b) {
  for (const auto& pair_a : a) {
    const string& key_a = pair_a.first;
    const double value_a = pair_a.second;
    // If the key is not present, this will initialize it to zero.
    (*b)[key_a] += value_a;
  }
}

std::map<string, double> GetSum(const std::map<string, double>& a,
                                const std::map<string, double>& b) {
  std::map<string, double> result = a;
  IncrementMap(b, &result);
  return result;
}

std::map<string, double> MapValues(const std::map<string, double>& input,
                                   const std::function<double(double)>& mapFn) {
  std::map<string, double> result;
  for (const auto& pair : input) {
    result[pair.first] = mapFn(pair.second);
  }
  return result;
}

std::map<string, double> IntMapToDoubleMap(
    const std::map<string, int64>& int_map) {
  std::map<string, double> result;
  for (const auto& pair : int_map) {
    result[pair.first] = pair.second;
  }
  return result;
}

}  // namespace data_validation
}  // namespace tensorflow
