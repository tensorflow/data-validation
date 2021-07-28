# Version 1.2.0

## Major Features and Improvements

*   Added statistics/generators/mutual_information.py. It estimates AMI using a
    knn estimation. It differs from sklearn_mutual_information.py in that this
    supports multivalent features/labels (by encoding) and multivariate
    features/labels. The plan is to deprecate sklearn_mutual_information.py in
    the future.
*   Fixed NonStreamingCustomStatsGenerator to respect max_batches_per_partition.

## Bug Fixes and Other Changes

*   Depends on 'scikit-learn>=0.23,<0.24' ("mutual-information" extra only)
*   Depends on 'scipy>=1.5,<2' ("mutual-information" extra only)
*   Depends on `apache-beam[gcp]>=2.31,<3`.
*   Depends on `tensorflow-metadata>=1.2,<1.3`.
*   Depends on `tfx-bsl>=1.2,<1.3`.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A

