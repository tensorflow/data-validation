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

#include "tensorflow_data_validation/anomalies/path.h"

#include <algorithm>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "re2/re2.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace data_validation {
namespace {

// This matches a standard step.
// Standard steps include any proto steps (extensions and regular fields).
// Standard steps either begin or end with parentheses with none on the inside
// or they have no parentheses, dot, or single quotes.
static const LazyRE2 kStandardStep = {R"((\([^()]*\))|([^().']+))",
                                      RE2::Latin1};

// This matches a serialized step with quotes.
// This begins and ends with a single quote, and all internal single quotes are
// doubled.
static const LazyRE2 kSerializedWithQuotes = {"'(('')|[^'])*'", RE2::Latin1};

// This matches a serialized step and the next dot.
// Note that this is the union of the above two regexes followed by a dot.
static const LazyRE2 kSerializedStepAndDot = {
    R"(((('(('')|[^'])*')|(\([^()]*\))|([^()'.]+))\.))", RE2::Latin1};

// Returns true if:
// str is nonempty and has no ".", "(", ")", or "'", OR:
// str starts with "(", ends with ")", and has no "(" or ")" in the interior.
bool IsStandardStep(const string& str) {
  return RE2::FullMatch(str, *kStandardStep);
}

string SerializeStep(const string& str) {
  if (IsStandardStep(str)) {
    return str;
  }
  // Double any single quotes in the string, and encapsulate with single quotes.
  return absl::StrCat("'", absl::StrReplaceAll(str, {{"'", "''"}}), "'");
}

// Deserialize a step in-place.
// If the step is in the standard format, do nothing.
// Otherwise, remove beginning and ending quote and replace pairs of single
// quotes with single quotes.
tensorflow::Status DeserializeStep(string* to_modify) {
  if (IsStandardStep(*to_modify)) {
    return Status::OK();
  }

  // A legal serialized string here begins and ends with a single quote and
  // has had all interior single quotes replaced with double quotes.
  if (!RE2::FullMatch(*to_modify, *kSerializedWithQuotes)) {
    return errors::InvalidArgument("Not a valid serialized step: ", *to_modify);
  }
  // Remove the first and last quote
  const absl::string_view quotes_removed(to_modify->data() + 1,
                                         to_modify->size() - 2);

  // Replace each pair of quotes remaining with a single quote.
  *to_modify = absl::StrReplaceAll(quotes_removed, {{"''", "'"}});
  return Status::OK();
}

// Find the next step delimiter.
// See absl/strings/str_split.h
// If the next step is not valid in some way,
// return absl::string_view(text.end(), 0)
struct StepDelimiter {
  absl::string_view Find(absl::string_view text, size_t pos) {
    if (pos >= text.size()) {
      return absl::string_view(text.end(), 0);
    }
    absl::string_view remaining_string = text.substr(pos);
    absl::string_view solution;
    // Regex captures a serialized step followed by a dot.
    // Note that this only captures the step if it is not the last one. Since
    // on no match we just have the rest of the string be a step, we are OK.
    // Note that solution is a view into a view into the original text argument.
    // Match(...,&solution, 1) will return 1 submatch (i.e. the substring of
    // text matching the regular expression).
    if (kSerializedStepAndDot->Match(remaining_string, 0,
                                     remaining_string.size(), RE2::ANCHOR_START,
                                     &solution, 1) &&
        solution.data() != nullptr) {
      // solution now contains the step and the dot after the step.
      // Returns a string_view of text that is equal to ".".
      return solution.substr(solution.size() - 1);
    }
    return absl::string_view(text.end(), 0);
  }
};
}  // namespace

Path::Path(const tensorflow::metadata::v0::Path& p)
    : step_(p.step().begin(), p.step().end()) {}

// Part of the implementation of Compare().
bool Path::Equals(const Path& p) const {
  if (p.step_.size() != step_.size()) {
    return false;
  }
  return std::equal(step_.begin(), step_.end(), p.step_.begin());
}

bool Path::Less(const Path& p) const {
  return std::lexicographical_compare(step_.begin(), step_.end(),
                                      p.step_.begin(), p.step_.end());
}

int Path::Compare(const Path& p) const {
  if (Equals(p)) {
    return 0;
  }
  return Less(p) ? -1 : +1;
}

bool operator==(const Path& a, const Path& b) { return a.Compare(b) == 0; }

bool operator<(const Path& a, const Path& b) { return a.Compare(b) < 0; }

bool operator>(const Path& a, const Path& b) { return a.Compare(b) > 0; }

bool operator>=(const Path& a, const Path& b) { return a.Compare(b) >= 0; }

bool operator<=(const Path& a, const Path& b) { return a.Compare(b) <= 0; }

bool operator!=(const Path& a, const Path& b) { return a.Compare(b) != 0; }

string Path::Serialize() const {
  const string separator = ".";
  std::vector<string> serialized_steps;
  for (const string& step : step_) {
    serialized_steps.push_back(SerializeStep(step));
  }
  return absl::StrJoin(serialized_steps, separator);
}

tensorflow::metadata::v0::Path Path::AsProto() const {
  tensorflow::metadata::v0::Path path;
  for (const string& step : step_) {
    path.add_step(step);
  }
  return path;
}

// Deserializes a string created with Serialize().
// Note: for any path p:
// p==Path::Deserialize(p.Serialize())
tensorflow::Status Path::Deserialize(absl::string_view str, Path* result) {
  result->step_.clear();
  if (str.empty()) {
    return Status::OK();
  }
  result->step_ = absl::StrSplit(str, StepDelimiter());
  for (string& step : result->step_) {
    TF_RETURN_IF_ERROR(DeserializeStep(&step));
  }
  return Status::OK();
}

Path Path::GetChild(absl::string_view last_step) const {
  std::vector<string> new_steps(step_.begin(), step_.end());
  new_steps.push_back(string(last_step));
  return Path(std::move(new_steps));
}

std::pair<string, Path> Path::PopHead() const {
  return {step_[0], Path({step_.begin() + 1, step_.end()})};
}

void PrintTo(const Path& p, std::ostream* o) {
  (*o) << p.Serialize();
}

}  // namespace data_validation
}  // namespace tensorflow
