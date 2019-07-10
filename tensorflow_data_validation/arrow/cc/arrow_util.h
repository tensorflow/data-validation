// Copyright 2019 Google LLC
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

// Note: ALL the functions declared here will be SWIG wrapped into
// pywrap_tensorflow_data_validation.
#ifndef TENSORFLOW_DATA_VALIDATION_ARROW_CC_ARROW_UTIL_H_
#define TENSORFLOW_DATA_VALIDATION_ARROW_CC_ARROW_UTIL_H_
#include <memory>

#include "Python.h"

// Flattens a ListArray by one level. Logically this means concatenating all
// the lists `list_array`, ignoring nulls.
// For example [[1,2,3], [], None, [4,5]] => [1,2,3,4,5].
// This is implemented natively as ListArray.flatten() in pyarrow 0.12.0.
// TODO(zhuo): Remove once Arrow used by TFDV is upgraded to >= 0.12.0.
PyObject* TFDV_Arrow_FlattenListArray(PyObject* py_list_array);

// Get lengths of lists in `list_array` in an int32 array.
// Note that null and empty list both are of length 0 and the returned array
// does not have any null element.
// For example [[1,2,3], [], None, [4,5]] => [3, 0, 0, 2]
PyObject* TFDV_Arrow_ListLengthsFromListArray(PyObject* py_list_array);

// Returns a int32 array of the same length as flattened `list_array`.
// returned_array[i] == j means i-th element in flattened `list_array`
// came from j-th list in `list_array`.
// For example [[1,2,3], [], None, [4,5]] => [0, 0, 0, 3, 3]
PyObject* TFDV_Arrow_GetFlattenedArrayParentIndices(PyObject* py_list_array);

// Returns a uint8 array of the same length as `array`.
// returned_array[i] == True iff array[i] is null.
// Note that this returned array can be converted to a numpy bool array
// copy-free.
PyObject* TFDV_Arrow_GetArrayNullBitmapAsByteArray(PyObject* array);

// Returns the total byte size of a BinaryArray (note that StringArray is a
// subclass of that so is also accepted here) i.e. the length of the
// concatenation of all the binary strings in the list), in a Python Long.
// It might be a good idea to make this a pyarrow API.
PyObject* TFDV_Arrow_GetBinaryArrayTotalByteSize(PyObject* py_binary_array);

// Get counts of values in the array. Returns a struct array <values, counts>.
PyObject* TFDV_Arrow_ValueCounts(PyObject* array);

#endif  // TENSORFLOW_DATA_VALIDATION_ARROW_CC_ARROW_UTIL_H_

