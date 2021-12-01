# Version 1.5.0

## Major Features and Improvements

*   N/A

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

