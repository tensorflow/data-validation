# Version 1.8.0

## Major Features and Improvements

*   From this version we will be releasing python 3.9 wheels.

## Bug Fixes and Other Changes

*   Adds `get_statistics_html` to the public API.
*   Fixes several incorrect type annotations.
*   Schema inference handles derived features.
*   `StatsOptions.to_json` now raises an error if it encounters unsupported
    options.
*   Depends on `apache-beam[gcp]>=2.38,<3`.
*   Depends on
    `tensorflow>=1.15.5,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3`.
*   Depends on `tensorflow-metadata>=1.8.0,<1.9.0`.
*   Depends on `tfx-bsl>=1.8.0,<1.9.0`.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A

