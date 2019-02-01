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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow_data_validation/anomalies/test_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow_metadata/proto/v0/path.pb.h"

namespace tensorflow {
namespace data_validation {
namespace {
using testing::ParseTextProtoOrDie;

MATCHER_P(EqualsPath, path,
          absl::StrCat((negation ? "doesn't equal" : "equals"),
                       path.Serialize())) {
  return path.Compare(arg) == 0;
}

TEST(Path, Constructor) {
  EXPECT_EQ("a.b.c", Path({"a", "b", "c"}).Serialize());
  EXPECT_EQ("", Path().Serialize());
  EXPECT_EQ("a.b.c", Path(ParseTextProtoOrDie<tensorflow::metadata::v0::Path>(
                              R"(step: [ "a", "b", "c" ])"))
                         .Serialize());
}

TEST(Path, AsProto) {
  const Path p({"a", "b", "c"});
  Path result(p.AsProto());
  EXPECT_THAT(result, EqualsPath(p)) << "result: " << result.Serialize();
}

// If compare identifies things as -1, 0, and 1 correctly, the rest of the
// methods can be tested with three examples.
TEST(Path, Compare) {
  EXPECT_EQ(1, Path({"a", "b", "c"}).Compare(Path({"a", "b"})));
  EXPECT_EQ(-1, Path({"a", "b"}).Compare(Path({"a", "b", "c"})));
  EXPECT_EQ(0, Path({"a", "b", "c"}).Compare(Path({"a", "b", "c"})));
  EXPECT_EQ(-1, Path({"a", "b", "c"}).Compare(Path({"a", "d", "c"})));
  EXPECT_EQ(1, Path({"a", "d", "c"}).Compare(Path({"a", "b", "c"})));
  EXPECT_EQ(0, Path().Compare(Path()));
}

// See TEST(Path, Compare) above.
TEST(Path, Less) {
  EXPECT_FALSE(Path({"a", "b", "c"}) < Path({"a", "b"}));
  EXPECT_TRUE(Path({"a", "b"}) < Path({"a", "b", "c"}));
  EXPECT_FALSE(Path({"a", "b", "c"}) < Path({"a", "b", "c"}));
}

// See TEST(Path, Compare) above.
TEST(Path, GreaterOrEqual) {
  EXPECT_TRUE(Path({"a", "b", "c"}) >= Path({"a", "b"}));
  EXPECT_FALSE(Path({"a", "b"}) >= Path({"a", "b", "c"}));
  EXPECT_TRUE(Path({"a", "b", "c"}) >= Path({"a", "b", "c"}));
}

// See TEST(Path, Compare) above.
TEST(Path, Greater) {
  EXPECT_TRUE(Path({"a", "b", "c"}) > Path({"a", "b"}));
  EXPECT_FALSE(Path({"a", "b"}) > Path({"a", "b", "c"}));
  EXPECT_FALSE(Path({"a", "b", "c"}) > Path({"a", "b", "c"}));
}

// See TEST(Path, Compare) above.
TEST(Path, LessOrEqual) {
  EXPECT_FALSE(Path({"a", "b", "c"}) <= Path({"a", "b"}));
  EXPECT_TRUE(Path({"a", "b"}) <= Path({"a", "b", "c"}));
  EXPECT_TRUE(Path({"a", "b", "c"}) <= Path({"a", "b", "c"}));
}

// See TEST(Path, Compare) above.
TEST(Path, Equal) {
  EXPECT_FALSE(Path({"a", "b", "c"}) == Path({"a", "b"}));
  EXPECT_FALSE(Path({"a", "b"}) == Path({"a", "b", "c"}));
  EXPECT_TRUE(Path({"a", "b", "c"}) == Path({"a", "b", "c"}));
}

// See TEST(Path, Compare) above.
TEST(Path, NotEqual) {
  EXPECT_TRUE(Path({"a", "b", "c"}) != Path({"a", "b"}));
  EXPECT_TRUE(Path({"a", "b"}) != Path({"a", "b", "c"}));
  EXPECT_FALSE(Path({"a", "b", "c"}) != Path({"a", "b", "c"}));
}

TEST(Path, Serialize) {
  EXPECT_EQ("a.'.b'.'''c'''", Path({"a", ".b", "'c'"}).Serialize());
  EXPECT_EQ("a.(b'.d).'((c)'", Path({"a", "(b'.d)", "((c)"}).Serialize());
  EXPECT_EQ("''", Path({""}).Serialize());
  EXPECT_EQ("", Path().Serialize());
}

TEST(Path, Deserialize) {
  std::vector<Path> paths_to_check = {Path({"a", ".b", "'c'"}),
                                      Path({"a", "(b'.d)", "((c)"}), Path({""}),
                                      Path()};
  for (const Path& path : paths_to_check) {
    Path result;
    TF_ASSERT_OK(Path::Deserialize(path.Serialize(), &result))
        << "Failed on " << path.Serialize() << "!=" << result.Serialize();
    EXPECT_THAT(result, EqualsPath(path)) << "result: " << result.Serialize();
  }
}

// If a path has steps that have quotes that didn't need to be quoted,
// Deserialize works anyway.
TEST(Path, DeserializeSillyQuotes) {
  Path no_silly_quotes;
  TF_ASSERT_OK(Path::Deserialize("'a'.'b'.'(c'')'", &no_silly_quotes));
  EXPECT_EQ("a.b.(c')", no_silly_quotes.Serialize());
}

// If a path has steps that have quotes that didn't need to be quoted,
// Deserialize works anyway.
TEST(Path, DeserializeBad) {
  const std::vector<string> bad_serializations = {
      "a'", "'a", "(b", "c'd", "'c'd'", "''cd'", "'c'''d'"};
  for (const string& bad : bad_serializations) {
    Path dummy;
    tensorflow::Status status = Path::Deserialize(bad, &dummy);
    EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT)
        << "Deserialize did not fail on " << bad;
  }
}

TEST(Path, GetParent) {
  EXPECT_EQ("a.b", Path({"a", "b", "c"}).GetParent().Serialize());
  EXPECT_EQ("a", Path({"a", "b"}).GetParent().Serialize());
}
TEST(Path, GetChild) {
  EXPECT_EQ("a.b", Path({"a"}).GetChild("b").Serialize());
  EXPECT_EQ("a", Path().GetChild("a").Serialize());
}

TEST(Path, size) {
  EXPECT_EQ(3, Path({"a", "b", "c"}).size());
  EXPECT_EQ(0, Path().size());
}
TEST(Path, empty) {
  EXPECT_EQ(false, Path({"a", "b", "c"}).empty());
  EXPECT_EQ(true, Path().empty());
}

TEST(Path, GetLastStep) {
  EXPECT_EQ("c", Path({"a", "b", "c"}).last_step());
  EXPECT_EQ("a", Path({"a"}).last_step());
}

TEST(Path, PopHead) {
  EXPECT_THAT(Path({"foo", "rest", "of", "path"}).PopHead(),
              ::testing::Pair("foo", Path({"rest", "of", "path"})));
  EXPECT_THAT(Path({"foo"}).PopHead(), ::testing::Pair("foo", Path()));
}

TEST(Path, PrintTo) {
  Path p({"a", "b", "c"});
  std::ostringstream os;
  PrintTo(p, &os);
  EXPECT_EQ(p.Serialize(), os.str());
}

}  // namespace
}  // namespace data_validation
}  // namespace tensorflow
