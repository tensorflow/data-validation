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

#include "tensorflow_data_validation/anomalies/test_schema_protos.h"

#include "tensorflow_data_validation/anomalies/test_util.h"

namespace tensorflow {
namespace data_validation {
namespace testing {

using ::tensorflow::metadata::v0::Schema;

Schema GetTestAllTypesMessage() {
  return ParseTextProtoOrDie<Schema>(R"(
  feature {
  name: "optional_bool"
  value_count {
    min: 1
    max: 1
  }
  type: INT
  int_domain {
    min: 0
    max: 1
  }
  presence {
    min_count: 1
  }
}
feature {
  name: "optional_enum"
  value_count {
    min: 1
    max: 1
  }
  type: BYTES
  domain: "FirstSixStates"
  presence {
    min_count: 1
  }
}
feature {
  name: "optional_float"
  value_count {
    min: 1
    max: 1
  }
  type: FLOAT
  presence {
    min_count: 1
  }
}
feature {
  name: "optional_int32"
  value_count {
    min: 1
    max: 1
  }
  type: INT
  int_domain {
    min: -2147483648
    max: 2147483647
  }
  presence {
    min_count: 1
  }
}
feature {
  name: "optional_int64"
  value_count {
    min: 1
    max: 1
  }
  type: INT
  presence {
    min_count: 1
  }
}
feature {
  name: "optional_string"
  value_count {
    min: 1
    max: 1
  }
  type: BYTES
  presence {
    min_count: 1
  }
}
feature {
  name: "optional_uint32"
  value_count {
    min: 1
    max: 1
  }
  type: INT
  int_domain {
    min: 0
    max: 4294967295
  }
  presence {
    min_count: 1
  }
}
feature {
  name: "repeated_bool"
  value_count {
    min: 1
  }
  type: INT
  int_domain {
    min: 0
    max: 1
  }
  presence {
    min_count: 1
  }
}
feature {
  name: "repeated_enum"
  value_count {
    min: 1
  }
  type: BYTES
  domain: "FirstSixStates"
  presence {
    min_count: 1
  }
}
feature {
  name: "repeated_float"
  value_count {
    min: 1
  }
  type: FLOAT
  presence {
    min_count: 1
  }
}
feature {
  name: "repeated_int32"
  value_count {
    min: 1
  }
  type: INT
  int_domain {
    min: -2147483648
    max: 2147483647
  }
  presence {
    min_count: 1
  }
}
feature {
  name: "repeated_int64"
  value_count {
    min: 1
  }
  type: INT
  presence {
    min_count: 1
  }
}
feature {
  name: "repeated_string"
  value_count {
    min: 1
  }
  type: BYTES
  presence {
    min_count: 1
  }
}
feature {
  name: "repeated_uint32"
  value_count {
    min: 1
  }
  type: INT
  int_domain {
    min: 0
    max: 4294967295
  }
  presence {
    min_count: 1
  }
}
string_domain {
  name: "FirstSixStates"
  value: "ALABAMA"
  value: "ALASKA"
  value: "ARIZONA"
  value: "ARKANSAS"
  value: "CALIFORNIA"
  value: "COLORADO"
})");
}

Schema GetAnnotatedFieldsMessage() {
  return ParseTextProtoOrDie<Schema>(R"(
      string_domain {
        name: "MyAnnotatedEnum"
        value: "4"
        value: "5"
        value: "6"
      }
      feature {
        name: "annotated_enum"
        presence: {min_count: 1} value_count {min: 1 max: 1}
        type: BYTES
        domain: "MyAnnotatedEnum"
      }
      feature {
        name: "big_int64"
        presence: {min_count: 1} value_count {min: 1}
        type: INT
        int_domain {
          min: 65
        }
      }
      feature {
        name: "bool_with_false"
        presence: {min_count: 1} value_count {min: 1 max: 1}
        type: BYTES
        bool_domain {
          false_value: "my_false"
        }
      }
      feature {
        name: "bool_with_true"
        presence: {min_count: 1} value_count {min: 1 max: 1}
        type: BYTES
        bool_domain {
          true_value: "my_true"
        }
      }
      feature {
        name: "bool_with_true_false"
        presence: {min_count: 1} value_count {min: 1 max: 1}
        type: BYTES
        bool_domain {
          true_value: "my_true"
          false_value: "my_false"
        }
      }
      feature {
        name: "few_int64"
        presence: {min_count: 1} value_count {min: 1 max: 3}
        type: INT
      }
      feature {
        name: "float_very_common"
        presence: {min_count: 1 min_fraction: 0.5} value_count {min: 1 max: 1}
        type: FLOAT
      }
      feature {
        name: "float_with_bounds"
        presence: {min_count: 1} value_count {min: 1 max: 1}
        type: FLOAT
        float_domain: { min: -3 max: 5 }
      }
      feature {
        name: "ignore_this"
        lifecycle_stage: DEPRECATED
        presence: {min_count: 1} value_count {min: 1}
        type: BYTES
      }
      feature {
        name: "small_int64"
        presence: {min_count: 1} value_count {min: 1}
        type: INT
        int_domain {
          max: 123
        }
      }
      feature {
        name: "string_int32"
        presence: {min_count: 1} value_count {min: 1 max: 1}
        type: BYTES
        int_domain {
          min: -2147483648
          max: 2147483647
        }
      }
      feature {
        name: "string_int64"
        presence: {min_count: 1} value_count {min: 1 max: 1}
        type: BYTES
        int_domain {
        }
      }
      feature {
        name: "string_uint32"
        presence: {min_count: 1} value_count {min: 1 max: 1}
        type: BYTES
        int_domain {
          min: 0
          max: 4294967295
        }
      }
)");
}

Schema GetTestSchemaAlone() {
  return ParseTextProtoOrDie<Schema>(R"(
      string_domain {
        name: "MyAloneEnum"
        value: "4"
        value: "5"
        value: "6"
        value: "ALONE_BUT_NORMAL"
      }
      feature {
        name: "annotated_enum"
        presence: {min_count: 1} value_count {min: 1 max: 1}
        type: BYTES
        domain: "MyAloneEnum"
      }
      feature {
        name: "big_int64"
        presence: {min_count: 1} value_count {min: 1}
        type: INT
        int_domain {
          min: 65
        }
      }
      feature {
        name: "bool_with_false"
        presence: {min_count: 1} value_count {min: 1 max: 1}
        type: BYTES
        bool_domain {
          false_value: "my_false"
        }
      }
      feature {
        name: "bool_with_true"
        presence: {min_count: 1} value_count {min: 1 max: 1}
        type: BYTES
        bool_domain {
          true_value: "my_true"
        }
      }
      feature {
        name: "bool_with_true_false"
        presence: {min_count: 1} value_count {min: 1 max: 1}
        type: BYTES
        bool_domain {
          true_value: "my_true"
          false_value: "my_false"
        }
      }
      feature {
        name: "few_int64"
        presence: {min_count: 1} value_count {min: 1 max: 3}
        type: INT
      }
      feature {
        name: "ignore_this"
        lifecycle_stage: DEPRECATED
        presence: {min_count: 1} value_count {min: 1}
        type: BYTES
      }
      feature {
        name: "small_int64"
        presence: {min_count: 1} value_count {min: 1}
        type: INT
        int_domain {
          max: 123
        }
      }
      feature {
        name: "string_int32"
        presence: {min_count: 1} value_count {min: 1 max: 1}
        type: BYTES
        int_domain {
          min: -2147483648
          max: 2147483647
        }
      }
      feature {
        name: "string_int64"
        presence: {min_count: 1} value_count {min: 1 max: 1}
        type: BYTES
        int_domain {
        }
      }
      feature {
        name: "string_uint32"
        presence: {min_count: 1} value_count {min: 1 max: 1}
        type: BYTES
        int_domain {
          min: 0
          max: 4294967295
        }
      }
)");
}

}  // namespace testing
}  // namespace data_validation
}  // namespace tensorflow
