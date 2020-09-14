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
