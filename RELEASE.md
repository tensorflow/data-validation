# Version 1.7.0

## Major Features and Improvements

*   Adds the `DetectFeatureSkew` PTransform to the public API, which can be used
    to detect feature skew between training and serving examples.
*   Uses sketch-based top-k/uniques in TFDV inmemory mode.

## Bug Fixes and Other Changes

*   Fixes a bug in load_statistics that would cause failure when reading binary
    protos.
*   Depends on `pyfarmhash>=0.2,<0.4`.
*   Depends on
    `tensorflow>=1.15.5,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3`.
*   Depends on `tensorflow-metadata>=1.7.0,<1.8.0`.
*   Depends on `tfx-bsl>=1.7.0,<1.8.0`.
*   Depends on `apache-beam[gcp]>=2.36,<3`.
*   Updated the documentation for CombinerStatsGenerator to clarify that the
    first accumulator passed to merge_accumulators may be modified.
*   Added compression type detection when reading csv header.
*   Detection of invalid utf8 strings now works regardless of relative frequency.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A

