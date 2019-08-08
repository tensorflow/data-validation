// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_DATA_VALIDATION_ARROW_CC_DECODED_EXAMPLES_TO_ARROW_H_
#define TENSORFLOW_DATA_VALIDATION_ARROW_CC_DECODED_EXAMPLES_TO_ARROW_H_

#include "Python.h"

// Converts a list of decoded examples (each is a
// Dict[Union[bytes, unicode], Union[None, np.ndarray]]) to an Arrow Table.
// The result table has M rows and N columns where M is the number of examples
// in the list and N is the number of unique features in the examples.
// Each column is either a ListArray<int64|float|binary> or a NullArray.
//
// None and missing feature handling:
//   - if a feature's value is None in an example, then its corresponding column
//     in the result table will have a null at the corresponding position.
//   - if a feature's value is always None across all the examples in the input
//     list, then its corresponding column in the result table will be a
//     NullArray.
//   - if an example does not contain a feature (in the universe of features),
//     then the column of that feature will have a null at the corresponding
//     position.
//
// Raises exceptions when:
//   - the input is not a list, or the element in the list is not Dict[
//     str, Union[None, np.ndarray]]
//   - any keys to the dicts are neither python bytes nor python unicode object.
//   - any ndarrays as value in the dicts are not of expected type (
//     only np.int64, np.float32 and np.object (for bytes feature) are expected)
//   - for np.object ndarrays, if any element is not a python bytes object.
//   - some dicts don't agree on the type of the values of the same feature.

PyObject* TFDV_Arrow_DecodedExamplesToTable(PyObject* list_of_decoded_examples);

#endif  // TENSORFLOW_DATA_VALIDATION_ARROW_CC_DECODED_EXAMPLES_TO_ARROW_H_
