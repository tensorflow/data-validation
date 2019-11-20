<!-- mdlint off(HEADERS_TOO_MANY_H1) -->
# Current version (not yet released; still in development)

## Major Features and Improvements

* Started depending on the CSV parsing / type inferring utilities provided
  by `tfx-bsl` (since tfx-bsl 0.15.2). This also brings performance improvements
  to the CSV decoder (~2x faster in decoding. Type inferring performance is not
  affected).


## Bug Fixes and Other Changes

* Exclude examples in which the entire sparse feature is missing when
  calculating sparse feature statistics.
* Validate min_examples_count dataset constraint.
* Document the schema fields, statistics fields, and detection condition for
  each anomaly type that TFDV detects.
* Handle null array in cross feature stats generator.
* Depends on `tensorflow-metadata>=0.15.1,<0.16`.

## Breaking Changes

* Changed the behavior regarding to statistics over CSV data:

  - Previously, if a CSV column was mixed with integers and empty strings, FLOAT
    statistics will be collected for that column. A change was made so INT
    statistics would be collected instead.

* Removed `csv_decoder.DecodeCSVToDict` as `Dict[str, np.ndarray]` had no longer
  been the internal data representation any more since 0.14.

## Deprecations

# Release 0.15.0

## Major Features and Improvements

* Generate statistics for sparse features.
* Directly convert a batch of tf.Examples to Arrow tables. Avoids conversion of
  tf.Example to intermediate Dict representation.

## Bug Fixes and Other Changes

* Generate statistics for the weight feature.
* Support validation and schema inference from sliced statistics that include
  the default slice (validation/inference will be done using the default slice
  statistics).
* Avoid flattening null arrays.
* Set `weighted_num_examples` field in the statistics proto if a weight
  feature is specified.
* Replace DecodedExamplesToTable with a Python implementation.
* Building TFDV from source does not need pyarrow anymore.
* Depends on `apache-beam[gcp]>=2.16,<3`.
* Depends on `six>=1.12,<2`.
* Depends on `scikit-learn>=0.18,<0.22`.
* Depends on `tfx-bsl>=0.15,<0.16`.
* Depends on `tensorflow-metadata>=0.15,<0.16`.
* Depends on `tensorflow-transform>=0.15,<0.16`.
* Depends on `tensorflow>=1.15,<3`.
  * Starting from 1.15, package
    `tensorflow` comes with GPU support. Users won't need to choose between
    `tensorflow` and `tensorflow-gpu`.
  * Caveat: `tensorflow` 2.0.0 is an exception and does not have GPU
    support. If `tensorflow-gpu` 2.0.0 is installed before installing
    `tensorflow-data-validation`, it will be replaced with `tensorflow` 2.0.0.
    Re-install `tensorflow-gpu` 2.0.0 if needed.

## Breaking Changes

## Deprecations

# Release 0.14.1

## Major Features and Improvements

* Add support for custom schema transformations when inferring schema.

## Bug Fixes and Other Changes

* Fix incorrect file hashes in the TFDV wheel.
* Fix DOMException when embedding visualization in iframe.

## Breaking Changes

## Deprecations

# Release 0.14.0

## Major Features and Improvements

* Performance improvement due to optimizing inner loops.
* Add support for time semantic domain related statistics.
* Performance improvement due to batching accumulators before merging.
* Add utility method `validate_examples_in_tfrecord`, which identifies anomalous
  examples in TFRecord files containing TFExamples and generates statistics for
  those anomalous examples.
* Add utility method `validate_examples_in_csv`, which identifies anomalous
  examples in CSV files and generates statistics for those anomalous examples.
* Add fast TF example decoder written in C++.
* Make `BasicStatsGenerator` to take arrow table as input. Example batches are
  converted to Apache Arrow tables internally and we are able to make use of
  vectorized numpy functions. Improved performance of BasicStatsGenerator
  by ~40x.
* Make `TopKUniquesStatsGenerator` and `TopKUniquesCombinerStatsGenerator` to
  take arrow table as input.
* Add `update_schema` API which updates the schema to conform to statistics.
* Add support for validating changes in the number of examples between the
  current and previous spans of data (using the existing `validate_statistics`
  function).
* Support building a manylinux2010 compliant wheel in docker.
* Add support for cross feature statistics.

## Bug Fixes and Other Changes

* Expand unit test coverage.
* Update natural language stats generator to generate stats if actual ratio
  equals `match_ratio`.
* Use `__slots__` in accumulators.
* Fix overflow warning when generating numeric stats for large integers.
* Set max value count in schema when the feature has same valency, thereby
  inferring shape for multivalent required features.
* Fix divide by zero error in natural language stats generator.
* Add `load_anomalies_text` and `write_anomalies_text` utility functions.
* Define ReasonFeatureNeeded proto.
* Add support for Windows OS.
* Make semantic domain stats generators to take arrow column as input.
* Fix error in number of missing examples and total number of examples
  computation.
* Make FeaturesNeeded serializable.
* Fix memory leak in fast example decoder.
* Add `semantic_domain_stats_sample_rate` option to compute semantic domain
  statistics over a sample.
* Increment refcount of None in fast example decoder.
* Add `compression_type` option to `generate_statistics_from_*` methods.
* Add link to SysML paper describing some technical details behind TFDV.
* Add Python types to the source code.
* Make`GenerateStatistics` generate a DatasetFeatureStatisticsList containing a
  dataset with num_examples == 0 instead of an empty proto if there are no
  examples in the input.
* Depends on `absl-py>=0.7,<1`
* Depends on `apache-beam[gcp]>=2.14,<3`
* Depends on `numpy>=1.16,<2`.
* Depends on `pandas>=0.24,<1`.
* Depends on `pyarrow>=0.14.0,<0.15.0`.
* Depends on `scikit-learn>=0.18,<0.21`.
* Depends on `tensorflow-metadata>=0.14,<0.15`.
* Depends on `tensorflow-transform>=0.14,<0.15`.

## Breaking Changes

* Change `examples_threshold` to `values_threshold` and update documentation to
  clarify that counts are of values in semantic domain stats generators.
* Refactor IdentifyAnomalousExamples to remove sampling and output
  (anomaly reason, example) tuples.
* Rename `anomaly_proto` parameter in anomalies utilities to `anomalies` to
  make it more consistent with proto and schema utilities.
* `FeatureNameStatistics` produced by `GenerateStatistics` is now identified
  by its `.path` field instead of the `.name` field. For example:

  ```
  feature {
    name: "my_feature"
  }
  ```
  becomes:

  ```
  feature {
    path {
      step: "my_feature"
    }
  }
  ```
* Change `validate_instance` API to accept an Arrow table instead of a Dict.
* Change `GenerateStatistics` API to accept Arrow tables as input.

## Deprecations

# Release 0.13.1

## Major Features and Improvements

## Bug Fixes and Other Changes

* Modify validation logic to raise `SCHEMA_MISSING_COLUMN` anomaly when
  observing a feature with no stats (was still broken, now fixed).

## Breaking Changes

## Deprecations

# Release 0.13.0

## Major Features and Improvements

* Use joblib to exploit multiprocessing when computing statistics over a pandas
  dataframe.
* Add support for semantic domain related statistics (natural language, image),
  enabled by `StatsOptions.enable_semantic_domain_stats`.
* Python 3.5 is supported.

## Bug Fixes and Other Changes

* Expand unit test coverage.
* Modify validation logic to raise `SCHEMA_MISSING_COLUMN` anomaly when
  observing a feature with no stats.
* Add utility functions `write_stats_text` and `load_stats_text` to write and
  load DatasetFeatureStatisticsList protos.
* Avoid using multiprocessing by default when generating statistics over a
  dataframe.
* Depends on `joblib>=0.12,<1`.
* Depends on `tensorflow-transform>=0.13,<0.14`.
* Depends on `tensorflow-metadata>=0.12.1,<0.14`.
* Requires pre-installed `tensorflow>=1.13.1,<2`.
* Depends on `apache-beam[gcp]>=2.11,<3`.
* Depends on `absl>=0.1.6,<1`.

## Breaking Changes

## Deprecations

# Release 0.12.0

## Major Features and Improvements

* Add support for computing statistics over slices of data.
* Performance improvement due to optimizing inner loops.
* Add support for generating statistics from a pandas dataframe.
* Performance improvement due to pre-allocating tf.Example in
  TFExampleDecoder.
* Performance improvement due to merging common stats generator, numeric stats
  generator and string stats generator as a single basic stats generator.
* Performance improvement due to merging top-k and uniques generators.
* Add a `validate_instance` function, which checks a single example for
  anomalies.
* Add a utility method `get_statistics_html`, which returns HTML that can be
  used for Facets visualization outside of a notebook.
* Add support for schema inference of semantic domains.
* Performance improvement on statistics computation over a pandas dataframe.

## Bug Fixes and Other Changes

* Use constant '__BYTES_VALUE__' in the statistics proto to represent a bytes
  value which cannot be decoded as a utf-8 string.
* Introduced CombinerFeatureStatsGenerator, a specialized interface for
  combiners that do not require cross-feature computations.
* Expand unit test coverage.
* Add optional frequency threshold that allows keeping only the most frequent
  values that are present in a minimum number of examples.
* Add optional desired batch size that allows specification of the number of
  examples to include in each batch.
* Depends on `numpy>=1.14.5,<2`.
* Depends on `protobuf>=3.6.1,<4`.
* Depends on `apache-beam[gcp]>=2.10,<3`.
* Depends on `tensorflow-metadata>=0.12.1,<0.13`.
* Depends on `scikit-learn>=0.18,<1`.
* Depends on `IPython>=5.0`.
* Requires pre-installed `tensorflow>=1.12,<2`.
* Revise example notebook and update it to be able to run in Colab and Jupyter.

## Breaking changes
* Represent batch as a list of ndarrays instead of ndarrays of ndarrays.
* Modify decoders to return ndarrays of type numpy.float32 for FLOAT features.

## Deprecations

# Release 0.11.0

## Major Features and Improvements

* Add option to infer feature types from schema when generating statistics over
  CSV data.
* Add utility method `set_domain` to set the domain of a feature in the schema.
* Add option to compute weighted statistics by providing a weight feature.
* Add a PTransform for decoding TF examples.
* Add utility methods `write_schema_text` and `load_schema_text` to write and
  load the schema protocol buffer.
* Add option to compute statistics over a sample.
* Optimize performance of statistics computation (~2x improvement on benchmark
  datasets).

## Bug Fixes and Other Changes

* Depends on `apache-beam[gcp]>=2.8,<3`.
* Depends on `tensorflow-transform>=0.11,<0.12`.
* Depends on `tensorflow-metadata>=0.9,<0.10`.
* Fix bug in clearing oneof domain\_info field in Feature proto.
* Fix overflow error for large integers by casting them to STRING type.
* Added API docs.

## Breaking changes

* Requires pre-installed `tensorflow>=1.11,<2`.
* Make tf.Example decoder to represent a feature with no value list as a
  missing value (None).
* Make StatsOptions as a class.

## Deprecations

# Release 0.9.0

* Initial release of TensorFlow Data Validation.
