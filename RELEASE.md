# Version 1.13.0

## Major Features and Improvements

*   Introduces a Schema option `HistogramSelection` to allow numeric drift/skew
    calculations to use QUANTILES histograms, which are more robust to outliers.

## Bug Fixes and Other Changes

*  Rename `statistics_io_impl` and `default_record_sink` (not part of public API).
*  Update the minimum Bazel version required to build TFDV to 5.3.0.
*  Depends on `numpy~=1.22.0`.
*  Depends on `pyfarmhash>=0.2.2,<0.4`.
*  Depends on `tensorflow>=2.12.0,<2.13`.
*  Depends on `protobuf>=3.20.3,<5`.
*  Depends on `tfx-bsl>=1.13.0,<1.14.0`.
*  Depends on `tensorflow-metadata>=1.13.1,<1.14.0`.

## Known Issues

*   N/A

## Breaking Changes

* Jensen-Shannon divergence now treats NaN values as always contributing to
  higher drift score.

## Deprecations

*   Deprecated python 3.7 support.

