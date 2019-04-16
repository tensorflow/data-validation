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

#ifndef TENSORFLOW_DATA_VALIDATION_CODERS_CC_FAST_EXAMPLE_DECODER_H_
#define TENSORFLOW_DATA_VALIDATION_CODERS_CC_FAST_EXAMPLE_DECODER_H_

#include <Python.h>

namespace tensorflow {
namespace data_validation {

// Parses serialized tf.example and decodes it into Dict[str, np.ndarray]
// Returns NULL and raises ValueError in the following cases:
// a) if the input serialized proto is not of bytes type
// b) if the input serialized proto cannot be converted to string type
// c) if the input cannot be parsed as a tf.Example proto
// d) if the result ndarray cannot be added to the Dict.
PyObject* TFDV_DecodeExample(PyObject* serialized_proto);

}  // namespace data_validation
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_VALIDATION_CODERS_CC_FAST_EXAMPLE_DECODER_H_
