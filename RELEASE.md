# Version 0.27.0

## Major Features and Improvements

*   Performance improvement to `BasicStatsGenerator`.

## Bug Fixes and Other Changes

*   Added a `compact()` and `setup()` interface to `CombinerStatsGenerator`,
    `CombinerFeatureStatsWrapperGenerator`, `BasicStatsGenerator`,
    `CompositeStatsGenerator`, and `ConstituentStatsGenerator`.
*   Stopped depending on `tensorflow-transform`.
*   Depends on `apache-beam[gcp]>=2.27,<3`.
*   Depends on `pyarrow>=1,<3`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,<3`.
*   Depends on `tensorflow-metadata>=0.27,<0.28`.
*   Depends on `tfx-bsl>=0.27,<0.28`.

## Known Issues

*   N/A

## Breaking changes

*   N/A

## Deprecations

*   `tfdv.DecodeCSV` and `tfdv.DecodeTFExample` are deprecated. Use
    `tfx_bsl.public.tfxio.CsvTFXIO` and `tfx_bsl.public.tfxio.TFExampleRecord`
    instead.
