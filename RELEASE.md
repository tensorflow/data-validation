# Version 1.14.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Bumped the Ubuntu version on which TFX-BSL is tested to 20.04 (previously
    was 16.04).
*   Use @platforms instead of @bazel_tools//platforms to specify constraints in
    OSS build.
*   Depends on `pyarrow>=10,<11`.
*   Depends on `apache-beam>=2.47,<3`.
*   Depends on `numpy>=1.22.0`.
*   Depends on `tensorflow>=2.13.0,<3`.

## Known Issues

*   N/A

## Breaking Changes

*  Moves some non-public arrow_util functions to TFX-BSL.
*  Changes SkewPair proto to store tf.Examples in serialized format.

## Deprecations

*   N/A

