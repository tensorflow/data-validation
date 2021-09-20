# Version 1.3.0

## Major Features and Improvements

*   N/A

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
