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

#ifndef TENSORFLOW_DATA_VALIDATION_ANOMALIES_MAP_UTIL_H_
#define TENSORFLOW_DATA_VALIDATION_ANOMALIES_MAP_UTIL_H_

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data_validation {

// Returns true if and only if the given container contains the given key.
template <typename Container, typename Key>
bool ContainsKey(const Container& container, const Key& key) {
  return container.find(key) != container.end();
}

// Adds the values in the map.
double SumValues(const std::map<string, double>& input);

// Gets the keys from the map. The order of the keys is the same as in
// the map.
std::vector<string> GetKeysFromMap(const std::map<string, double>& input);

// Gets the values from the map. The order of the values is the same as in
// the map.
std::vector<double> GetValuesFromMap(const std::map<string, double>& input);

// Normalizes the values, such that the sum of the values are 1.
// If the values sum to zero, return the input map.
std::map<string, double> Normalize(const std::map<string, double>& input);

// Gets the difference of the values of two maps. Values that are not
// present are treated as zero.
std::map<string, double> GetDifference(const std::map<string, double>& a,
                                       const std::map<string, double>& b);

// Gets the sum of the values of two maps. Values that are not
// present are treated as zero.
std::map<string, double> GetSum(const std::map<string, double>& a,
                                const std::map<string, double>& b);

// Increments one map by another. Values that are not
// present are treated as zero.
void IncrementMap(const std::map<string, double>& a,
                  std::map<string, double>* b);

// Applies a function to all the values in the map.
std::map<string, double> MapValues(const std::map<string, double>& input,
                                   const std::function<double(double)>& mapFn);

// Cast the values from int64 to double. Notice that this might lose some
// information.
std::map<string, double> IntMapToDoubleMap(
    const std::map<string, int64>& int_map);

}  // namespace data_validation
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_VALIDATION_ANOMALIES_MAP_UTIL_H_
