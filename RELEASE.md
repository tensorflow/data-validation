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

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   Deprecated python 3.6 support.

