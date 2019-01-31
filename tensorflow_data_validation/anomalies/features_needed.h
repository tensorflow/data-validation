#ifndef TENSORFLOW_DATA_VALIDATION_ANOMALIES_FEATURES_NEEDED_H_
#define TENSORFLOW_DATA_VALIDATION_ANOMALIES_FEATURES_NEEDED_H_

#include "tensorflow_data_validation/anomalies/path.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

namespace tensorflow {
namespace data_validation {

// TODO(martinz): consider making a protobuf.
// TODO(martinz): consider adding an environment.
// TODO(martinz): consider adding a LifecycleStage.
struct ReasonFeatureNeeded {
  // If there is an issue in creating the field, the comment should help
  // explain why.
  // Example: "This is needed for transform XYZ (see /A/B/C)"
  string comment;
};

using FeaturesNeeded = std::map<Path, std::vector<ReasonFeatureNeeded>>;

}  // namespace data_validation
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_VALIDATION_ANOMALIES_FEATURES_NEEDED_H_
