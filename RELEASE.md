# Version 1.0.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Increased the threshold beyond which a string feature value is considered
    "large" by the experimental sketch-based top-k/unique generator to 1024.
*   Added normalized AMI to sklearn mutual information generator.
*   Depends on `apache-beam[gcp]>=2.29,<3`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,<3`.
*   Depends on `tensorflow-metadata>=1.0,<1.1`.
*   Depends on `tfx-bsl>=1.0,<1.1`.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*  Removed the following deprecated symbols. Their deprecation was announced
   in 0.30.0.
  - `tfdv.validate_instance`
  - `tfdv.lift_stats_generator`
  - `tfdv.partitioned_stats_generator`
  - `tfdv.get_feature_value_slicer`
*  Removed parameter `compression_type` in
   `tfdv.generate_statistics_from_tfrecord`
