#include "tensorflow_data_validation/anomalies/schema_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"

namespace tensorflow {
namespace data_validation {

namespace {
using tensorflow::metadata::v0::AnomalyInfo;

// Since this method only has nine possible inputs, it is easiest to test them
// all directly.
TEST(MaxSeverity, MaxSeverity) {
  EXPECT_EQ(AnomalyInfo::UNKNOWN,
            MaxSeverity(AnomalyInfo::UNKNOWN, AnomalyInfo::UNKNOWN));
  EXPECT_EQ(AnomalyInfo::WARNING,
            MaxSeverity(AnomalyInfo::UNKNOWN, AnomalyInfo::WARNING));
  EXPECT_EQ(AnomalyInfo::ERROR,
            MaxSeverity(AnomalyInfo::UNKNOWN, AnomalyInfo::ERROR));
  EXPECT_EQ(AnomalyInfo::WARNING,
            MaxSeverity(AnomalyInfo::WARNING, AnomalyInfo::UNKNOWN));
  EXPECT_EQ(AnomalyInfo::WARNING,
            MaxSeverity(AnomalyInfo::WARNING, AnomalyInfo::WARNING));
  EXPECT_EQ(AnomalyInfo::ERROR,
            MaxSeverity(AnomalyInfo::WARNING, AnomalyInfo::ERROR));
  EXPECT_EQ(AnomalyInfo::ERROR,
            MaxSeverity(AnomalyInfo::ERROR, AnomalyInfo::UNKNOWN));
  EXPECT_EQ(AnomalyInfo::ERROR,
            MaxSeverity(AnomalyInfo::ERROR, AnomalyInfo::WARNING));
  EXPECT_EQ(AnomalyInfo::ERROR,
            MaxSeverity(AnomalyInfo::ERROR, AnomalyInfo::ERROR));
}

}  // namespace

}  // namespace data_validation
}  // namespace tensorflow
