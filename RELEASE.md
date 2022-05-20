<!-- mdlint off(HEADERS_TOO_MANY_H1) -->

# Current Version (Still in Development)

## Major Features and Improvements

## Bug Fixes and Other Changes

*   Depends on `tensorflow>=1.15.5,<2` or `tensorflow>=2.9,<3`

## Known Issues

## Breaking Changes

## Deprecations

# Version 1.8.0

## Major Features and Improvements

*   From this version we will be releasing python 3.9 wheels.

## Bug Fixes and Other Changes

*   Adds `get_statistics_html` to the public API.
*   Fixes several incorrect type annotations.
*   Schema inference handles derived features.
*   `StatsOptions.to_json` now raises an error if it encounters unsupported
    options.
*   Depends on `apache-beam[gcp]>=2.38,<3`.
*   Depends on `tensorflow-metadata>=1.8.0,<1.9.0`.
*   Depends on `tfx-bsl>=1.8.0,<1.9.0`.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.7.0

## Major Features and Improvements

*   Adds the `DetectFeatureSkew` PTransform to the public API, which can be used
    to detect feature skew between training and serving examples.
*   Uses sketch-based top-k/uniques in TFDV inmemory mode.

## Bug Fixes and Other Changes

*   Fixes a bug in load_statistics that would cause failure when reading binary
    protos.
*   Depends on `pyfarmhash>=0.2,<0.4`.
*   Depends on
    `tensorflow>=1.15.5,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3`.
*   Depends on `tensorflow-metadata>=1.7.0,<1.8.0`.
*   Depends on `tfx-bsl>=1.7.0,<1.8.0`.
*   Depends on `apache-beam[gcp]>=2.36,<3`.
*   Updated the documentation for CombinerStatsGenerator to clarify that the
    first accumulator passed to merge_accumulators may be modified.
*   Added compression type detection when reading csv header.
*   Detection of invalid utf8 strings now works regardless of relative frequency.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.6.0

## Major Features and Improvements

*   Introduces a convenience wrapper for handling indexed access to statistics
    protos.
*   String features are checked for UTF-8 validity, and the number of invalid
    strings is reported as invalid_utf8_count.

## Bug Fixes and Other Changes

*   Depends on `numpy>=1.16,<2`.
*   Depends on `absl-py>=0.9,<2.0.0`.
*   Depends on
    `tensorflow>=1.15.5,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,<3`.
*   Depends on `tensorflow-metadata>=1.6.0,<1.7.0`.
*   Depends on `tfx-bsl>=1.6.0,<1.7.0`.
*   Depends on `apache-beam[gcp]>=2.35,<3`.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.5.0

## Major Features and Improvements

## Bug Fixes and Other Changes

*   BasicStatsGenerator is now responsible for setting the global num_examples.
    This field will no longer be populated at the DatasetFeatureStatistics level
    if default generators are disabled.
*   Depends on `apache-beam[gcp]>=2.34,<3`.
*   Depends on
    `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,<3`.
*   Depends on `tensorflow-metadata>=1.5.0,<1.6.0`.
*   Depends on `tfx-bsl>=1.5.0,<1.6.0`.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.4.0

## Major Features and Improvements

*   Float features can now be analyzed as categorical for the purposes of top-k
    and unique count using experimental sketch based generators.
*   Support SQL based slicing in TFDV. This would enable slicing (using SQL) in
    TFX OSS and Dataflow environments. SQL based slicing is currently not
    supported on Windows.

## Bug Fixes and Other Changes

*   Variance calculations have been updated to be more numerically stable for
    large datasets or large magnitude numeric data.
*   When running per-example validation against a schema, output of
    validate_examples_in_tfrecord and validate_examples_in_csv now optionally
    return samples of anomalous examples.
*   Changes to source code ensures that it can now work with `pyarrow>=3`.
*   Add `load_anomalies_binary` utility function.
*   Merge two accumulators at a time instead of batching.
*   BasicStatsGenerator is now responsible for setting
    FeatureNameStatistics.Type. Previously it was possible for a top-k generator
    and BasicStatsGenerator to set different types for categorical numeric
    features with physical type STRING.
*   Depends on `pyarrow>=1,<6`.
*   Depends on `tensorflow-metadata>=1.4,<1.5`.
*   Depends on `tfx-bsl>=1.4,<1.5`.
*   PartitionedStatsFn can optionally provide their own PTransform to control
    how inputs are partitioned.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   Deprecated python 3.6 support.

# Version 1.3.0

## Major Features and Improvements

## Bug Fixes and Other Changes

*   Fixed bug in JensenShannonDivergence calculation affecting comparisons of
    histograms that each contain a single value.
*   Fixed bug in dataset constraints validation that caused failures with very
    large numbers of examples.
*   Fixed a bug wherein slicing on a feature missing from some batches could
    produce slice keys derived from a different feature.
*   Depends on `apache-beam[gcp]>=2.32,<3`.
*   Depends on
    `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,<3`.
*   Depends on `tfx-bsl>=1.3,<1.4`.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.2.0

## Major Features and Improvements

*   Added statistics/generators/mutual_information.py. It estimates AMI using a
    knn estimation. It differs from sklearn_mutual_information.py in that this
    supports multivalent features/labels (by encoding) and multivariate
    features/labels. The plan is to deprecate sklearn_mutual_information.py in
    the future.
*   Fixed NonStreamingCustomStatsGenerator to respect max_batches_per_partition.

## Bug Fixes and Other Changes

*   Switched from namedtuple to tfx_namedtuple in order to avoid pickling issues
    with PySpark.
*   Depends on 'scikit-learn>=0.23,<0.24' ("mutual-information" extra only)
*   Depends on 'scipy>=1.5,<2' ("mutual-information" extra only)
*   Depends on `apache-beam[gcp]>=2.31,<3`.
*   Depends on `tensorflow-metadata>=1.2,<1.3`.
*   Depends on `tfx-bsl>=1.2,<1.3`.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.1.1

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `google-cloud-bigquery>=1.28.0,<2.21`.
*   Depends on `tfx-bsl>=1.1.1,<1.2`.
*   Fixes error when using tfdv.experimental_get_feature_value_slicer with
    pandas==1.3.0.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.1.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Optimized certain stats generators that needs to materialize the input
    RecordBatches.
*   Depends on `protobuf>=3.13,<4`.
*   Depends on `tensorflow-metadata>=1.1,<1.2`.
*   Depends on `tfx-bsl>=1.1,<1.2`.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.0.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Increased the threshold beyond which a string feature value is considered
    "large" by the experimental sketch-based top-k/unique generator to 1024.
*   Added normalized AMI to sklearn mutual information generator.
*   Depends on `apache-beam[gcp]>=2.29,<3`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,<3`.
*   Depends on `tensorflow-metadata>=1.0,<1.1`.
*   Depends on `tfx-bsl>=1.0,<1.1`.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*  Removed the following deprecated symbols. Their deprecation was announced
   in 0.30.0.
  - `tfdv.validate_instance`
  - `tfdv.lift_stats_generator`
  - `tfdv.partitioned_stats_generator`
  - `tfdv.get_feature_value_slicer`
*  Removed parameter `compression_type` in
   `tfdv.generate_statistics_from_tfrecord`

# Version 0.30.0

## Major Features and Improvements

*   This version is the last version before TFDV 1.0. Once 1.0, all the TFDV
    public APIs (i.e. symbols in the root `__init__.py`) will be subject to
    semantic versioning. We are deprecating some public APIs in this version
    and they will be removed in 1.0.

*   Sketch-based top-k/unique stats generator now is able to detect invalid
    utf-8 sequences / large texts and replace them with a placeholder.
    It will not suffer from memory issue usually caused by image / large text
    features in the data. Note that this generator is not by default used yet.
*   Added `StatsOptions.experimental_use_sketch_based_topk_uniques` which
    enables the sketch-based top-k/unique stats generator.

## Bug Fixes and Other Changes

*   Fixed bug in `display_schema` that caused domains not to be displayed.
*   Modified how `get_schema_dataframe` outputs numeric domains.
*   Anomalies previously (un)classified as UKNOWN_TYPE now trigger more specific
    anomaly types: INVALID_DOMAIN_SPECIFICATION and MULTIPLE_REASONS.
*   Depends on `tensorflow-metadata>=0.30,<0.31`.
*   Depends on `tfx-bsl>=0.30,<0.31`.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   `tfdv.LiftStatsGenerator` is going to be removed in the next version from
    the public API. To enable that generator,
    supply `StatsOptions.label_feature`
*   `tfdv.NonStreamingCustomStatsGenerator` is going to be removed in the next
    version from the public API. You may continue to import it from TFDV
    but it will not be subject to compatibility guarantees.
*   `tfdv.validate_instance` is going to be removed in the next
    version from the public API. You may continue to import it from TFDV
    but it will not be subject to compatibility guarantees.
*   Removed `tfdv.DecodeCSV`, `tfdv.DecodeTFExample` (deprecated in 0.27).
*   Removed `feature_whitelist` in `tfdv.StatsOptions` (deprecated in 0.28).
    Use `feature_allowlist` instead.
*   `tfdv.get_feature_value_slicer` is deprecated.
    `tfdv.experimental_get_feature_value_slicer` is introduced as a replacement.
    TFDV is likely to have a different slicing functionality post 1.0, which
    may not be compatible with the current slicers.
*   `StatsOptions.slicing_functions` is deprecated.
    `StatsOptions.experimental_slicing_functions` is introduced as a
    replacement.
*   `tfdv.WriteStatisticsToText` is removed (deprecated in 0.25.0).
*   Parameter `compression_type` in `tfdv.generate_statistics_from_tfrecord`
    is deprecated. The compression type is currently automatically determined.

# Version 0.29.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Added check for invalid min and max values for `values_counts` for nested
    features.
*   Bumped the mininum bazel version required to build TFDV to 3.7.2.
*   Depends on `absl-py>=0.9,<0.13`.
*   Depends on `tensorflow-metadata>=0.29,<0.30`.
*   Depends on `tfx-bsl>=0.29,<0.30`.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.28.0

## Major Features and Improvements

*   Add anomaly detection for max bytes size for images.

## Bug Fixes and Other Changes

*   Depends on `numpy>=1.16,<1.20`.
*   Fixed a bug that affected all CombinerFeatureStatsGenerators.
*   Allow for `bytes` type in `get_feature_value_slicer` in addition to `Text`
    and `int`.
*   Fixed a bug that caused TFDV to improperly infer a fixed shape when
    `tfdv.infer_schema` and `tfdv.update_schema` were called with
    `infer_feature_shape=True`.
*   Deprecated parameter `infer_feature_shape` of function `tfdv.update_schema`.
    If a schema feature has a pre-defined shape, `tfdv.update_schema` will
    always validate it. Otherwise, it will not try to add a shape.
*   Deprecated `tfdv.StatsOptions.feature_whitelist` and added
    `feature_allowlist` as a replacement. The former will be removed in the next
    release.
*   Added `get_schema_dataframe` and `get_anomalies_dataframe` utility
    functions.
*   Depends on `apache-beam[gcp]>=2.28,<3`.
*   Depends on `tensorflow-metadata>=0.28,<0.29`.
*   Depends on `tfx-bsl>=0.28.1,<0.29`.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.27.0

## Major Features and Improvements

*   Performance improvement to `BasicStatsGenerator`.

## Bug Fixes and Other Changes

*   Added a `compact()` and `setup()` interface to `CombinerStatsGenerator`,
    `CombinerFeatureStatsWrapperGenerator`, `BasicStatsGenerator`,
    `CompositeStatsGenerator`, and `ConstituentStatsGenerator`.
*   Stopped depending on `tensorflow-transform`.
*   Depends on `apache-beam[gcp]>=2.27,<3`.
*   Depends on `pyarrow>=1,<3`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,<3`.
*   Depends on `tensorflow-metadata>=0.27,<0.28`.
*   Depends on `tfx-bsl>=0.27,<0.28`.

## Known Issues

*   N/A

## Breaking changes

*   N/A

## Deprecations

*   `tfdv.DecodeCSV` and `tfdv.DecodeTFExample` are deprecated. Use
    `tfx_bsl.public.tfxio.CsvTFXIO` and `tfx_bsl.public.tfxio.TFExampleRecord`
    instead.

# Version 0.26.1

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `apache-beam[gcp]>=2.25,!=2.26.*,<2.29`.

## Known Issues

*   N/A

## Breaking changes

*   N/A

## Deprecations

*   N/A

# Version 0.26.0

## Major Features and Improvements

*  Added support for per-feature example weights which allows associating each
   column its specific weight column. See the `per_feature_weight_override`
   parameter in `StatsOptions.__init__`.

## Bug Fixes and Other Changes

*   Newly added LifecycleStage.DISABLED is now exempt from validation (similar
    to LifecycleStage.DEPRECATED, etc).
*   Fixed a bug where TFDV blindly trusts the claim type in the provided schema.
    TFDV now computes the stats according to the actual type of the data, and
    only when the actual type matches the claim in the schema will it compute
    type-specific stats (e.g. categorical ints).
*   Added an option to control whether to add default stats generators when
    `tfdv.GenerateStatistics()`.
*   Started using a new quantiles computation routine that does not depend on
    TF. This could potentially increase the performance of TFDV under certain
    workloads.
*   Extending schema_util to support sematic domains.
*   Moving natural_language_stats_generator to
    natural_language_domain_inferring_stats_generator and creating a new
    natural_language_stats_generator based on the fields of
    natural_language_domain.
*   Providing vocab_utils to assist in opening / loading vocabulary files.
*   A SchemaDiff will be reported upon J-S skew/drift.
*   Fixed a bug in FLOAT_TYPE_SMALL_FLOAT anomaly message.
*   Depends on `apache-beam[gcp]>=2.25,!=2.26.*,<3`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.4.*,<3`.
*   Depends on `tensorflow-metadata>=0.26,<0.27`.
*   Depends on `tensorflow-transform>=0.26,<0.27`.
*   Depends on `tfx-bsl>=0.26,<0.27`.

## Known Issues

*   N/A

## Breaking changes

*   N/A

## Deprecations

*   N/A

# Version 0.25.0

## Major Features and Improvements

*   Add support for detecting drift and distribution skew in numeric features.
*   `tfdv.validate_statistics` now also reports the raw measurements of
    distribution skew/drift (if any is done), regardless whether skew/drift is
    detected. The report is in the `drift_skew_info` of the `Anomalies` proto
    (return value of `validate_statistics`).
*   From this release TFDV will also be hosting nightly packages on
    https://pypi-nightly.tensorflow.org. To install the nightly package use the
    following command:

    ```
    pip install --extra-index-url https://pypi-nightly.tensorflow.org/simple tensorflow-data-validation
    ```

    Note: These nightly packages are unstable and breakages are likely to
    happen. The fix could often take a week or more depending on the complexity
    involved for the wheels to be available on the PyPI cloud service. You can
    always use the stable version of TFDV available on PyPI by running the
    command `pip install tensorflow-data-validation` .

## Bug Fixes and Other Changes

*   Added `tfdv.load_stats_binary` to load stats what were written using
    `tfdv.WriteStatisticsToText` (now `tfdv.WriteStatisticsToBinaryFile`).
*   Anomalies previously (un)classified as UKNOWN_TYPE now trigger more specific
    anomaly types: DOMAIN_INVALID_FOR_TYPE, UNEXPECTED_DATA_TYPE,
    FEATURE_MISSING_NAME, FEATURE_MISSING_TYPE, INVALID_SCHEMA_SPECIFICATION
*   Fixed a bug that `import tensorflow_data_validation` would fail if IPython
    is not installed. IPython is an optional dependency of TFDV.
*   Depends on `apache-beam[gcp]>=2.25,<3`.
*   Depends on `tensorflow-metadata>=0.25,<0.26`.
*   Depends on `tensorflow-transform>=0.25,<0.26`.
*   Depends on `tfx-bsl>=0.25,<0.26`.

## Known Issues

*   N/A

## Breaking Changes

*   `tfdv.WriteStatisticsToText` is renamed as
    `tfdv.WriteStatisticsToBinaryFile`. The former is still available but will
    be removed in a future release.

## Deprecations

*   N/A

# Version 0.24.1

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `apache-beam[gcp]>=2.24,<3`.
*   Depends on `tensorflow-transform>=0.24.1,<0.25`.
*   Depends on `tfx-bsl>=0.24.1,<0.25`.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.24.0

## Major Features and Improvements

*  You can now build the TFDV wheel with `python setup.py bdist_wheel`. Note:
  * If you want to build a manylinux2010 wheel you'll still need
    to use Docker.
  * Bazel is still required.
*  You can now build manylinux2010 TFDV wheel for Python 3.8.

## Bug Fixes and Other Changes

*   Support allowlist and denylist features in `tfdv.visualize_statistics`
    method.
*   Depends on `absl-py>=0.9,<0.11`.
*   Depends on `pandas>=1.0,<2`.
*   Depends on `protobuf>=3.9.2,<4`.
*   Depends on `tensorflow-metadata>=0.24,<0.25`.
*   Depends on `tensorflow-transform>=0.24,<0.25`.
*   Depends on `tfx-bsl>=0.24,<0.25`.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   Deprecated Py3.5 support.
*   Deprecated `sample_count` option in `tfdv.StatsOptions`. Use `sample_rate`
    option instead.

# Version 0.23.1

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `apache-beam[gcp]>=2.24,<3`.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   Deprecating python 3.5 support.

# Version 0.23.0

## Major Features and Improvements

*   Data validation is now able to handle arbitrarily nested arrow
    List/LargeList types. Schema entries for features with multiple nest levels
    describe the value count at each level in the value_counts field.
*   Add combiner stats generator to estimate top-K and uniques using Misra-Gries
    and K-Minimum Values sketches.

## Bug Fixes and Other Changes

*   Validate that enough supported images are present (if
    image_domain.minimum_supported_image_fraction is provided).
*   Stopped requiring avro-python3.
*   Depends on `apache-beam[gcp]>=2.23,<3`.
*   Depends on `pyarrow>=0.17,<0.18`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,<3`.
*   Depends on `tensorflow-metadata>=0.23,<0.24`.
*   Depends on `tensorflow-transform>=0.23,<0.24`.
*   Depends on `tfx-bsl>=0.23,<0.24`.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   Note: We plan to remove Python 3.5 support after this release.

# Version 0.22.2

## Major Features and Improvements

## Bug Fixes and Other Changes
*   Fixed a bug that affected tfx 0.22.0 to work with TFDV 0.22.1.
*   Depends on 'avro-python3>=1.8.1,<1.9.2' on Python 3.5 + MacOS

## Known Issues

## Breaking Changes

## Deprecations

# Version 0.22.1

## Major Features and Improvements

*   Statistics generation is now able to handle arbitrarily nested arrow
    List/LargeList types. Stats about the list elements' presence and valency
    are computed at each nest level, and stored in a newly added field,
    `valency_and_presence_stats` in `CommonStatistics`.

## Bug Fixes and Other Changes

*   Trigger DATASET_HIGH_NUM_EXAMPLES when a dataset has more than the specified
    limit on number of examples.
*   Fix bug in display_anomalies that prevented dataset-level anomalies from
    being displayed.
*   Trigger anomalies when a feature has a number of unique values that does not
    conform to the specified minimum/maximum.
*   Trigger anomalies when a float feature has unexpected Inf / -Inf values.
*   Depends on `apache-beam[gcp]>=2.22,<3`.
*   Depends on `pandas>=0.24,<2`.
*   Depends on `tensorflow-metadata>=0.22.2,<0.23.0`.
*   Depends on `tfx-bsl>=0.22.1,<0.23.0`.

## Known Issues

## Breaking Changes

## Deprecations

# Version 0.22.0

## Major Features and Improvements

## Bug Fixes and Other Changes

*   Crop values in natural language stats generator.
*   Switch to using PyBind11 instead of SWIG for wrapping C++ libraries.
*   CSV decoder support for multivalent columns by using tfx_bsl's decoder.
*   When inferring a schema entry for a feature, do not add a shape with dim = 0
    when min_num_values = 0.
*   Add utility methods `tfdv.get_slice_stats` to get statistics for a slice and
    `tfdv.compare_slices` to compare statistics of two slices using Facets.
*   Make `tfdv.load_stats_text` and `tfdv.write_stats_text` public.
*   Add PTransforms `tfdv.WriteStatisticsToText` and
    `tfdv.WriteStatisticsToTFRecord` to write statistics proto to text and
    tfrecord files respectively.
*   Modify `tfdv.load_statistics` to handle reading statistics from TFRecord and
    text files.
*   Added an extra requirement group `mutual-information`. As a result, barebone
    TFDV does not require `scikit-learn` any more.
*   Added an extra requirement group `visualization`. As a result, barebone TFDV
    does not require `ipython` any more.
*   Added an extra requirement group `all` that specifies all the extra
    dependencies TFDV needs. Use `pip install tensorflow-data-validation[all]`
    to pull in those dependencies.
*   Depends on `pyarrow>=0.16,<0.17`.
*   Depends on `apache-beam[gcp]>=2.20,<3`.
*   Depends on `ipython>=7,<8;python_version>="3"'.
*   Depends on `scikit-learn>=0.18,<0.24'.
*   Depends on `tensorflow>=1.15,!=2.0.*,<3`.
*   Depends on `tensorflow-metadata>=0.22.0,<0.23`.
*   Depends on `tensorflow-transform>=0.22,<0.23`.
*   Depends on `tfx-bsl>=0.22,<0.23`.

## Known Issues

*  (Known issue resolution) It is no longer necessary to use Apache Beam 2.17
   when running TFDV on Windows. The current release of Apache Beam will work.

## Breaking Changes

*   `tfdv.GenerateStatistics` now accepts a PCollection of `pa.RecordBatch`
    instead of `pa.Table`.
*   All the TFDV coders now output a PCollection of `pa.RecordBatch` instead of
    a PCollection of `pa.Table`.
*   `tfdv.validate_instances` and
    `tfdv.api.validation_api.IdentifyAnomalousExamples` now takes
    `pa.RecordBatch` as input instead of `pa.Table`.
*   The `StatsGenerator` interface (and all its sub-classes) now takes
    `pa.RecordBatch` as the input data instead of `pa.Table`.
*   Custom slicing functions now accepts a `pa.RecordBatch` instead of
    `pa.Table` as input and should output a tuple `(slice_key, record_batch)`.

## Deprecations

*   Deprecating Py2 support.

# Release 0.21.5

## Major Features and Improvements

*   Add `label_feature` to `StatsOptions` and enable `LiftStatsGenerator` when
    `label_feature` and `schema` are provided.
*   Add JSON serialization support for StatsOptions.

## Bug Fixes and Other Changes
*   Only requires `avro-python3>=1.8.1,!=1.9.2.*,<2.0.0` on Python 3.5 + MacOS

## Breaking Changes

## Deprecations

# Release 0.21.4

## Major Features and Improvements

*   Support visualizing feature value lift in facets visualization.

## Bug Fixes and Other Changes

*   Fix issue writing out string feature values in LiftStatsGenerator.
*   Requires 'apache-beam[gcp]>=2.17,<3'.
*   Requires 'tensorflow-transform>=0.21.1,<0.22'.
*   Requires 'tfx-bsl>=0.21.3,<0.22'.

## Breaking Changes

## Deprecations

# Release 0.21.2

## Major Features and Improvements

## Bug Fixes and Other Changes

*   Fix facets visualization.
*   Optimize LiftStatsGenerator for string features.
*   Make `_WeightedCounter` serializable.
*   Add support computing for weighted examples in LiftStatsGenerator.

## Breaking Changes

## Deprecations

*   `tfdv.TFExampleDecoder` has been removed. This legacy decoder converts
    serialized `tf.Example` to a dict of numpy arrays, which is the legacy
    input format (prior to Apache Arrow). TFDV has stopped accepting that format
    since 0.14. Use `tfdv.DecodeTFExample` instead.

# Release 0.21.1

## Major Features and Improvements

## Bug Fixes and Other Changes
* Do validation on weighted feature stats.
* During schema inference, skip features which are missing common stats. This
  makes schema inference work when the input stats are generated from some
  pre-existing, unknown schema.
* Fix facets visualization in Chrome >=M80.

## Known Issues

*   Running TFDV with Apache Beam 2.18 or 2.19 does not work on Windows. If you
    are using TFDV on Windows, use Apache Beam 2.17.

## Breaking Changes

## Deprecations

# Release 0.21.0

## Major Features and Improvements

* Started depending on the CSV parsing / type inferring utilities provided
  by `tfx-bsl` (since tfx-bsl 0.15.2). This also brings performance improvements
  to the CSV decoder (~2x faster in decoding. Type inferring performance is not
  affected).
* Compute bytes statistics for features of BYTES type. Avoid computing topk and
  uniques for such features.
* Added LiftStatsGenerator which computes lift between one feature (typically a
  label) and all other categorical features.

## Bug Fixes and Other Changes

*   Exclude examples in which the entire sparse feature is missing when
    calculating sparse feature statistics.
*   Validate min_examples_count dataset constraint.
*   Document the schema fields, statistics fields, and detection condition for
    each anomaly type that TFDV detects.
*   Handle null array in cross feature stats generator, top-k & uniques combiner
    stats generator, and sklearn mutual information generator.
*   Handle infinity in basic stats generator.
*   Set num_missing and num_examples correctly in the presence of sparse
    features.
*   Compute weighted feature stats for all weighted features declared in schema.
*   Enforce that mutual information is non-negative.
*   Depends on `tensorflow-metadata>=0.21.0,<0.22`.
*   Depends on `pyarrow>=0.15` (removed the upper bound as it is determined by
    `tfx-bsl`).
*   Depends on `tfx-bsl>=0.21.0,<0.22`
*   Depends on `apache-beam>=2.17,<3`
*   Validate that float feature does not contain NaNs (if disallow_nan is True).

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
