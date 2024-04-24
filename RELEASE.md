# Version 1.15.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   When computing cross feature statistics, skip configured crosses that
    include features of unsupported types (i.e., are not univalent numeric
    features).
*   Update the minimum Bazel version required to build TFDV to 6.1.0.
*   Modifies get_statistics_html() utility function to return a value indicating
    a dataset has no examples.
*   Outputs both a standard and a quantiles histogram for level N value list
    length statistics.
*   Add a `macos_arm64` config setting to the TFDV build file. NOTE: At this
    time, any M1 support for TFDV is experimental and untested.
*   Bumps the pybind11 version to 2.11.1.
*   Depends on `tensorflow~=2.15.0`.
*   Depends on `apache-beam[gcp]>=2.53.0,<3` for Python 3.11 and on 
    `apache-beam[gcp]>=2.47.0,<3` for 3.9 and 3.10.
*   Depends on `protobuf>=4.25.2,<5` for Python 3.11 and on `protobuf>3.20.3,<5`
    for 3.9 and 3.10.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   Deprecated python 3.8 support.
*   Deprecated Windows support.

