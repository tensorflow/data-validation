# Version 1.11.0

## Major Features and Improvements

*   This is the last version that supports TensorFlow 1.15.x. TF 1.15.x support
    will be removed in the next version. Please check the
    [TF2 migration guide](https://www.tensorflow.org/guide/migrate) to migrate
    to TF2.

*  Add a `custom_validate_statistics` function to the validation API, and
   support passing custom validations to `validate_statistics`. Note that
   custom validation is not supported on Windows.

## Bug Fixes and Other Changes

*   Fix bug in implementation of `semantic_domain_stats_sample_rate`.

*   Add beam metrics on string length

*   Determine whether to calculate string statistics based on the
    `is_categorical` field in the schema string domain.

*   Histograms counts should now be more accurate for distributions with few
    distinct values, or frequent individual values.

*   Nested list length histogram counts are no longer based on the number of
    values one up in the nested list hierarchy.

*   Support using jensen-shannon divergence to detect drift and skew for string
    and categorical features.

*   `get_drift_skew_dataframe` now includes a `threshold` column.

*   Adds support for NormalizedAbsoluteDifference comparator.

*   Depends on `tensorflow>=1.15.5,<2` or `tensorflow>=2.10,<3`
*   Depends on `joblib>=1.2.0`.

## Known Issues

*   N/A

## Breaking Changes

*   Histogram semantics are slightly changed, so that buckets include their
    upper bound instead of their lower bound. STANDARD histograms will no longer
    generate buckets that contain infinite and finite endpoints together.
*   Introduces StatsOptions.use_sketch_based_topk_uniques replacing
    experimental_use_sketch_based_topk_uniques. The latter option can still be
    written, but not read.

## Deprecations

*   N/A

