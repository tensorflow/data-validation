#include "tensorflow_data_validation/anomalies/schema_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"

namespace tensorflow {
namespace data_validation {

namespace {
int NumericalSeverity(tensorflow::metadata::v0::AnomalyInfo::Severity a) {
  switch (a) {
    case tensorflow::metadata::v0::AnomalyInfo::UNKNOWN:
      return 0;
    case tensorflow::metadata::v0::AnomalyInfo::WARNING:
      return 1;
    case tensorflow::metadata::v0::AnomalyInfo::ERROR:
      return 2;
    default:
      LOG(FATAL) << "Unknown severity: " << a;
  }
}
}  // namespace
// For internal use only.
tensorflow::metadata::v0::AnomalyInfo::Severity MaxSeverity(
    tensorflow::metadata::v0::AnomalyInfo::Severity a,
    tensorflow::metadata::v0::AnomalyInfo::Severity b) {
  return (NumericalSeverity(a) > NumericalSeverity(b)) ? a : b;
}

}  // namespace data_validation
}  // namespace tensorflow
