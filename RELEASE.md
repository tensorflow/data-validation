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

