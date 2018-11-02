#ifndef TENSORFLOW_DATA_VALIDATION_ANOMALIES_SCHEMA_UTIL_H_
#define TENSORFLOW_DATA_VALIDATION_ANOMALIES_SCHEMA_UTIL_H_

#include "tensorflow_metadata/proto/v0/anomalies.pb.h"

namespace tensorflow {
namespace data_validation {

// Returns the maximum (more serious) severity.
tensorflow::metadata::v0::AnomalyInfo::Severity MaxSeverity(
    tensorflow::metadata::v0::AnomalyInfo::Severity a,
    tensorflow::metadata::v0::AnomalyInfo::Severity b);

}  // namespace data_validation
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_VALIDATION_ANOMALIES_SCHEMA_UTIL_H_
