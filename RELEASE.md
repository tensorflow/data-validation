# Version 0.23.0

## Major Features and Improvements

*   Data validation is now able to handle arbitrarily nested arrow
    List/LargeList types. Schema entries for features with multiple nest levels
    describe the value count at each level in the value_counts field.
*   Add combiner stats generator to estimate top-K and uniques using Misra-Gries
    and K-Minimum Values sketches.

## Bug Fixes and Other Changes

*   Validate that enough supported images are present (if
    image_domain.minimum_supported_image_fraction is provided).
*   Stopped requiring avro-python3.
*   Depends on `apache-beam[gcp]>=2.23,<3`.
*   Depends on `pyarrow>=0.17,<0.18`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,<3`.
*   Depends on `tensorflow-metadata>=0.23,<0.24`.
*   Depends on `tensorflow-transform>=0.23,<0.24`.
*   Depends on `tfx-bsl>=0.23,<0.24`.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A
