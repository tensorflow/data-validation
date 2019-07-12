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

#include "tensorflow_data_validation/arrow/cc/merge.h"

#include <cstring>
#include <iostream>
#include <memory>

#include "arrow/python/init.h"
#include "arrow/python/pyarrow.h"
#include "arrow/array/concatenate.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/variant.h"
#include "arrow/api.h"
#include "arrow/visitor_inline.h"
#include "tensorflow_data_validation/arrow/cc/common.h"
#include "tensorflow_data_validation/arrow/cc/init_numpy.h"

namespace {
using ::arrow::AllocateEmptyBitmap;
using ::arrow::Array;
using ::arrow::ArrayData;
using ::arrow::ArrayVector;
using ::arrow::Buffer;
using ::arrow::Column;
using ::arrow::Concatenate;
using ::arrow::DataType;
using ::arrow::Field;
using ::arrow::Schema;
using ::arrow::Status;
using ::arrow::Table;
using ::arrow::Type;
using ::tensorflow::data_validation::ImportNumpy;
using ::tensorflow::data_validation::ImportPyArrow;

// If a column contains multiple chunks, concatenates those chunks into one and
// makes a new column out of it.
Status CompactColumn(const Column& column,
                     std::shared_ptr<Column>* compacted) {
  std::shared_ptr<Array> merged_data_array;
  RETURN_NOT_OK(Concatenate(column.data()->chunks(),
                            arrow::default_memory_pool(), &merged_data_array));
  *compacted = std::make_shared<Column>(column.field(), merged_data_array);
  return Status::OK();
}

// Compacts all the columns in `table`.
// TODO(zhuo): this function is no longer needed once Table.Compact() is
// available.
Status CompactTable(const Table& table, std::shared_ptr<Table>* compacted) {
  std::vector<std::shared_ptr<Column>> compacted_columns;
  for (int i = 0; i < table.num_columns(); ++i) {
    std::shared_ptr<Column> compacted_column;
    RETURN_NOT_OK(CompactColumn(*table.column(i), &compacted_column));
    compacted_columns.push_back(std::move(compacted_column));
  }
  *compacted = Table::Make(table.schema(), compacted_columns);
  return Status::OK();
}

// Makes arrays of nulls of various types.
class ArrayOfNullsMaker {
 public:
  ArrayOfNullsMaker(const std::shared_ptr<DataType>& type,
                    const int64_t num_nulls) {
    out_ = std::make_shared<ArrayData>(type, num_nulls, num_nulls);
  }

  Status Make(std::shared_ptr<ArrayData>* out) {
    RETURN_NOT_OK(arrow::VisitTypeInline(*out_->type, this));
    *out = std::move(out_);
    return Status::OK();
  }

  Status Visit(const arrow::NullType&) {
    return Status::OK();
  }

  Status Visit(const arrow::BooleanType&) {
    std::shared_ptr<Buffer> bitmap;
    RETURN_NOT_OK(AllocateNullBitmap(&bitmap));
    out_->buffers = {bitmap, bitmap};
    return Status::OK();
  }

  Status Visit(const arrow::FixedWidthType& fixed) {
    if (fixed.bit_width() % 8 != 0) {
      return Status::Invalid("Invalid bit width. Expected single byte entries");
    }
    std::shared_ptr<Buffer> null_bitmap;
    RETURN_NOT_OK(AllocateNullBitmap(&null_bitmap));
    std::shared_ptr<Buffer> data;
    RETURN_NOT_OK(
        AllocateBuffer(out_->null_count * fixed.bit_width() / 8, &data));
    out_->buffers = {null_bitmap, data};
    return Status::OK();
  }

  Status Visit(const arrow::BinaryType&) {
    return SetArrayDataForListAlike(/*type_has_data_buffer=*/true);
  }

  Status Visit(const arrow::ListType& l) {
    RETURN_NOT_OK(SetArrayDataForListAlike(/*type_has_data_buffer=*/false));
    out_->child_data.emplace_back();
    RETURN_NOT_OK(
        ArrayOfNullsMaker(l.value_type(), 0).Make(&out_->child_data.back()));
    return Status::OK();
  }

  // TODO(pachristopher): Remove external only tags after arrow 0.14.
  Status Visit(const arrow::FixedSizeListType& l) {
    return Status::NotImplemented(
        absl::StrCat("Make array of nulls: ", l.ToString()));
  }

  Status Visit(const arrow::StructType& s) {
    std::shared_ptr<Buffer> null_bitmap;
    RETURN_NOT_OK(AllocateNullBitmap(&null_bitmap));
    out_->buffers = {null_bitmap};
    for (int i = 0; i < s.num_children(); ++i) {
      std::shared_ptr<ArrayData> child_data;
      RETURN_NOT_OK(ArrayOfNullsMaker(s.child(i)->type(), out_->null_count)
                        .Make(&child_data));
      out_->child_data.push_back(std::move(child_data));
    }
    return Status::OK();
  }

  Status Visit(const arrow::DictionaryType& d) {
    return Status::NotImplemented(
        absl::StrCat("Make array of nulls: ", d.ToString()));
  }

  Status Visit(const arrow::UnionType& u) {
    return Status::NotImplemented(
        absl::StrCat("Make array of nulls: ", u.ToString()));
  }

  Status Visit(const arrow::ExtensionType& e) {
    return Status::NotImplemented(
        absl::StrCat("Make array of nulls: ", e.ToString()));
  }

 private:
  Status AllocateNullBitmap(std::shared_ptr<Buffer>* buf) const {
    RETURN_NOT_OK(AllocateEmptyBitmap(arrow::default_memory_pool(),
                                      out_->null_count, buf));
    return Status::OK();
  }

  Status AllocateNullOffsets(std::shared_ptr<Buffer>* buf) const {
    RETURN_NOT_OK(AllocateBuffer(arrow::default_memory_pool(),
                                 (out_->null_count + 1) * sizeof(int32_t),
                                 buf));
    // The offsets of an array of nulls are all 0.
    memset((*buf)->mutable_data(), 0, static_cast<size_t>((*buf)->size()));
    return Status::OK();
  }

  Status SetArrayDataForListAlike(const bool type_has_data_buffer) {
    std::shared_ptr<Buffer> null_bitmap;
    RETURN_NOT_OK(AllocateNullBitmap(&null_bitmap));
    std::shared_ptr<Buffer> offsets;
    RETURN_NOT_OK(AllocateNullOffsets(&offsets));
    out_->buffers = {null_bitmap, offsets};
    if (type_has_data_buffer) {
      out_->buffers.push_back(nullptr);
    }
    return Status::OK();
  }

  std::shared_ptr<ArrayData> out_;
};

// Makes an Array of `type` and `length` which contains all nulls.
// Only supports ListArray for now.
Status MakeArrayOfNulls(const std::shared_ptr<DataType>& type, int64_t length,
                        std::shared_ptr<Array>* result) {
  std::shared_ptr<ArrayData> array_data;
  RETURN_NOT_OK(ArrayOfNullsMaker(type, length).Make(&array_data));
  *result = arrow::MakeArray(array_data);
  return Status::OK();
}

// Represents a field (i.e. column name and type) in the merged table.
class FieldRep {
 public:
  FieldRep(std::shared_ptr<Field> field, const int64_t leading_nulls)
      : field_(std::move(field)) {
    if (leading_nulls > 0) {
      arrays_or_nulls_.push_back(leading_nulls);
    }
  }

  // Appends `column` to the column of the field in merged table.
  Status AppendColumn(const Column& column) {
    const auto& this_type = field_->type();
    const auto& other_type = column.type();
    if (other_type->id() == Type::NA) {
      AppendNulls(column.length());
      return Status::OK();
    }
    if (this_type->id() == Type::NA) {
      field_ = column.field();
    } else if (!this_type->Equals(other_type)) {
      return Status::Invalid(
          absl::StrCat("Trying to append a column of different type. ",
                       "Column name: ", column.name(),
                       " , Current type: ", this_type->ToString(),
                       " , New type: ", other_type->ToString()));
    }
    arrays_or_nulls_.insert(arrays_or_nulls_.end(),
                            column.data()->chunks().begin(),
                            column.data()->chunks().end());
    return Status::OK();
  }

  // Appends `num_nulls` nulls to the column of the field in the merged table.
  void AppendNulls(const int64_t num_nulls) {
    if (arrays_or_nulls_.empty() ||
        absl::holds_alternative<std::shared_ptr<Array>>(
            arrays_or_nulls_.back())) {
      arrays_or_nulls_.push_back(num_nulls);
    } else {
      absl::get<int64_t>(arrays_or_nulls_.back()) += num_nulls;
    }
  }

  const std::shared_ptr<Field>& field() const {
    return field_;
  }

  // Makes a merged array out of columns and nulls appended that can form
  // the column of the field in the merged table.
  Status ToMergedArray(std::shared_ptr<Array>* merged_array) const {
    if (field_->type()->id() == Type::NA) {
      arrow::NullBuilder null_builder;
      if (!arrays_or_nulls_.empty()) {
        RETURN_NOT_OK(null_builder.AppendNulls(
            absl::get<int64_t>(arrays_or_nulls_.back())));
      }
      RETURN_NOT_OK(null_builder.Finish(merged_array));
    } else {
      std::vector<std::shared_ptr<Array>> arrays_to_merge;
      for (const auto& array_or_num_nulls : arrays_or_nulls_) {
        std::shared_ptr<Array> array;
        if (absl::holds_alternative<int64_t>(array_or_num_nulls)) {
          RETURN_NOT_OK(MakeArrayOfNulls(
              field_->type(), absl::get<int64_t>(array_or_num_nulls), &array));
        } else {
          array = absl::get<std::shared_ptr<Array>>(array_or_num_nulls);
        }
        arrays_to_merge.push_back(std::move(array));
      }
      RETURN_NOT_OK(Concatenate(arrays_to_merge, arrow::default_memory_pool(),
                                merged_array));
    }
    return Status::OK();
  }

  std::shared_ptr<Field> field_;
  std::vector<absl::variant<std::shared_ptr<Array>, int64_t>> arrays_or_nulls_;
};

// Merges `tables` into one table, `merged`.
Status MergeTables(const std::vector<std::shared_ptr<Table>>& tables,
                   std::shared_ptr<Table>* merged) {
  absl::flat_hash_map<std::string, int> field_index_by_field_name;
  std::vector<FieldRep> field_rep_by_field_index;
  int64_t total_num_rows = 0;
  for (const auto& t : tables) {
    std::vector<bool> field_seen_by_field_index(field_rep_by_field_index.size(),
                                                false);
    for (int i = 0; i < t->schema()->num_fields(); ++i) {
      const auto& f = t->schema()->field(i);
      auto iter = field_index_by_field_name.find(f->name());
      if (iter == field_index_by_field_name.end()) {
        std::tie(iter, std::ignore) = field_index_by_field_name.insert(
            std::make_pair(f->name(), field_rep_by_field_index.size()));
        field_rep_by_field_index.emplace_back(f, total_num_rows);
        field_seen_by_field_index.push_back(true);
      }
      field_seen_by_field_index[iter->second] = true;
      FieldRep& field_rep = field_rep_by_field_index[iter->second];
      RETURN_NOT_OK(field_rep.AppendColumn(*t->column(i)));
    }
    for (int i = 0; i < field_seen_by_field_index.size(); ++i) {
      if (!field_seen_by_field_index[i]) {
        field_rep_by_field_index[i].AppendNulls(t->num_rows());
      }
    }
    total_num_rows += t->num_rows();
  }

  std::vector<std::shared_ptr<Field>> fields;
  ArrayVector merged_arrays;
  for (const FieldRep& field_rep : field_rep_by_field_index) {
    fields.push_back(field_rep.field());
    std::shared_ptr<Array> merged_array;
    RETURN_NOT_OK(field_rep.ToMergedArray(&merged_array));
    merged_arrays.push_back(std::move(merged_array));
  }
  *merged =
      Table::Make(std::make_shared<Schema>(std::move(fields)),
                  merged_arrays, /*num_rows=*/total_num_rows);
  return Status::OK();
}

// TODO(zhuo): This function is no longer needed once Table.Slice() is
// available.
std::shared_ptr<Table> SliceTable(const Table& table, int64_t offset,
                                  int64_t length) {
  const int num_columns = table.num_columns();
  std::vector<std::shared_ptr<Column>> sliced_columns(num_columns);
  for (int i = 0; i < num_columns; ++i) {
    sliced_columns[i] = table.column(i)->Slice(offset, length);
  }
  return Table::Make(table.schema(), sliced_columns, length);
}

Status SliceTableByRowIndices(const Table& table,
                              absl::Span<const int32_t> row_indices,
                              std::shared_ptr<Table>* out) {
  if (row_indices.empty()) {
    return Table::FromRecordBatches(table.schema(), {}, out);
  }

  if (row_indices.back() >= table.num_rows()) {
    return Status::Invalid("Row indices out of range.");
  }

  std::vector<std::shared_ptr<Table>> table_slices;
  // The following loop essentially turns consecutive indices into
  // ranges, and slice `table` by those ranges.
  for (int64_t begin = 0, end = 1; end <= row_indices.size(); ++end) {
    while (end < row_indices.size() &&
           row_indices[end] == row_indices[end - 1] + 1) {
      ++end;
    }
    // Verify that the row indices are sorted.
    if (end < row_indices.size() && row_indices[begin] >= row_indices[end]) {
      return Status::Invalid("Row indices are not sorted.");
    }
    table_slices.push_back(SliceTable(table, row_indices[begin], end - begin));
    begin = end;
  }
  std::shared_ptr<Table> concatenated;
  RETURN_NOT_OK(arrow::ConcatenateTables(table_slices, &concatenated));
  return CompactTable(*concatenated, out);
}

}  // namespace

PyObject* TFDV_Arrow_MergeTables(PyObject* list_of_tables) {
  ImportPyArrow();
  std::vector<std::shared_ptr<Table>> unwrapped_tables;
  if (!PyList_Check(list_of_tables)) {
    PyErr_SetString(PyExc_TypeError, "MergeTables expected a list.");
    return nullptr;
  }
  const Py_ssize_t list_size = PyList_Size(list_of_tables);
  if (list_size == 0) {
    PyErr_SetString(PyExc_ValueError, "MergeTables expected a non-empty list.");
    return nullptr;
  }
  for (int i = 0; i < list_size; ++i) {
    std::shared_ptr<Table> unwrapped;
    TFDV_RAISE_IF_NOT_OK(arrow::py::unwrap_table(
        PyList_GetItem(list_of_tables, i), &unwrapped));
    unwrapped_tables.push_back(std::move(unwrapped));
  }
  std::shared_ptr<Table> merged;
  TFDV_RAISE_IF_NOT_OK(MergeTables(unwrapped_tables, &merged));
  return arrow::py::wrap_table(merged);
}

PyObject* TFDV_Arrow_CompactTable(PyObject* table) {
  ImportPyArrow();
  std::shared_ptr<Table> unwrapped;
  TFDV_RAISE_IF_NOT_OK(arrow::py::unwrap_table(table, &unwrapped));
  std::shared_ptr<Table> compacted;
  TFDV_RAISE_IF_NOT_OK(CompactTable(*unwrapped, &compacted));
  return arrow::py::wrap_table(compacted);
}

PyObject* TFDV_Arrow_SliceTableByRowIndices(PyObject* table,
                                            PyObject* row_indices) {
  ImportPyArrow();
  ImportNumpy();
  std::shared_ptr<Table> unwrapped;
  TFDV_RAISE_IF_NOT_OK(arrow::py::unwrap_table(table, &unwrapped));
  PyArrayObject* row_indices_np;
  if (!PyArray_Check(row_indices)) {
    PyErr_SetString(
        PyExc_TypeError,
        "SliceTableByRowIndices expected row_indices to be a numpy array.");
    return nullptr;
  }

  row_indices_np = reinterpret_cast<PyArrayObject*>(row_indices);
  if (PyArray_TYPE(row_indices_np) != NPY_INT32 ||
      PyArray_NDIM(row_indices_np) != 1) {
    PyErr_SetString(PyExc_TypeError,
                    "SliceTableByRowIndices expected row_indices to be a 1-D "
                    "int32 numpy array.");
    return nullptr;
  }

  absl::Span<const int32_t> row_indices_span(
      static_cast<const int32_t*>(PyArray_DATA(row_indices_np)),
      PyArray_SIZE(row_indices_np));
  std::shared_ptr<Table> result;
  TFDV_RAISE_IF_NOT_OK(
      SliceTableByRowIndices(*unwrapped, row_indices_span, &result));
  return arrow::py::wrap_table(result);
}
