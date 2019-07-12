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
#ifndef THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ARROW_CC_MERGE_H_
#define THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ARROW_CC_MERGE_H_

#include "Python.h"

// Merges a list of arrow tables into one.
// The columns are concatenated (there will be only one chunk per column).
// Columns of the same name must be of the same type, or be a column of
// NullArrays.
// If a column in some tables are of type T, in some other tables are of
// NullArrays, the concatenated column is of type T, with nulls representing
// the rows from the table with NullArrays.
// If a column appears in some tables but not in some other tables, the
// concatenated column will contain nulls representing the rows from the table
// with that column missing.
PyObject* TFDV_Arrow_MergeTables(PyObject* list_of_tables);

// Compacts each column in `table` -- if the column consists of multiple chunks,
// concatenate them into a new column.
PyObject* TFDV_Arrow_CompactTable(PyObject* table);

// Collects rows in `row_indices` from `table` and form a new Table.
// The new Table is guaranteed to contain only one chunk.
// `row_indices` must be a 1-D int32 numpy array. And the indices in it
// must be sorted in ascending order.
PyObject* TFDV_Arrow_SliceTableByRowIndices(PyObject* table,
                                            PyObject* row_indices);

#endif  // THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ARROW_CC_MERGE_H_
