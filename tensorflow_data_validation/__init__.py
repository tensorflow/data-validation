# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Init module for TensorFlow Data Validation."""

# Import stats API.
from tensorflow_data_validation.api.stats_api import (
    GenerateStatistics,
    MergeDatasetFeatureStatisticsList,
    WriteStatisticsToBinaryFile,
    WriteStatisticsToRecordsAndBinaryFile,
    WriteStatisticsToTFRecord,
    default_sharded_output_suffix,
    default_sharded_output_supported,
)

# Import validation API.
from tensorflow_data_validation.api.validation_api import (
    DetectFeatureSkew,
    infer_schema,
    update_schema,
    validate_corresponding_slices,
    validate_statistics,
)

# Base classes for stats generators.
from tensorflow_data_validation.statistics.generators.stats_generator import (
    CombinerStatsGenerator,
    TransformStatsGenerator,
)

# Import stats options.
from tensorflow_data_validation.statistics.stats_options import StatsOptions

# Import FeaturePath.
from tensorflow_data_validation.types import FeaturePath

# Import anomalies utilities.
from tensorflow_data_validation.utils.anomalies_util import (
    load_anomalies_text,
    write_anomalies_text,
)

# Import display utilities.
from tensorflow_data_validation.utils.display_util import (
    compare_slices,
    display_anomalies,
    display_schema,
    get_confusion_count_dataframes,
    get_match_stats_dataframe,
    get_skew_result_dataframe,
    get_statistics_html,
    visualize_statistics,
)

# Import schema utilities.
from tensorflow_data_validation.utils.schema_util import (
    generate_dummy_schema_with_paths,
    get_domain,
    get_feature,
    load_schema_text,
    set_domain,
    write_schema_text,
)

# Import slicing utilities.
from tensorflow_data_validation.utils.slicing_util import (
    get_feature_value_slicer as experimental_get_feature_value_slicer,
)

# Import stats lib.
from tensorflow_data_validation.utils.stats_gen_lib import (
    generate_statistics_from_csv,
    generate_statistics_from_dataframe,
    generate_statistics_from_tfrecord,
)

# Import stats utilities.
from tensorflow_data_validation.utils.stats_util import (
    CrossFeatureView,
    DatasetListView,
    DatasetView,
    FeatureView,
    get_feature_stats,
    get_slice_stats,
    load_sharded_statistics,
    load_statistics,
    load_stats_binary,
    load_stats_text,
    write_stats_text,
)

# Import validation lib.
from tensorflow_data_validation.utils.validation_lib import (
    validate_examples_in_csv,
    validate_examples_in_tfrecord,
)

# Import version string.
from tensorflow_data_validation.version import __version__
