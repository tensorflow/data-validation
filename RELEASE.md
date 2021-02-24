<!-- mdlint off(HEADERS_TOO_MANY_H1) -->

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
