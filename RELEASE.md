<!-- mdlint off(HEADERS_TOO_MANY_H1) -->

# Version 0.22.0

## Major Features and Improvements

## Bug Fixes and Other Changes

*   Crop values in natural language stats generator.
*   Switch to using PyBind11 instead of SWIG for wrapping C++ libraries.
*   CSV decoder support for multivalent columns by using tfx_bsl's decoder.
*   When inferring a schema entry for a feature, do not add a shape with dim = 0
    when min_num_values = 0.
*   Add utility methods `tfdv.get_slice_stats` to get statistics for a slice and
    `tfdv.compare_slices` to compare statistics of two slices using Facets.
*   Make `tfdv.load_stats_text` and `tfdv.write_stats_text` public.
*   Add PTransforms `tfdv.WriteStatisticsToText` and
    `tfdv.WriteStatisticsToTFRecord` to write statistics proto to text and
    tfrecord files respectively.
*   Modify `tfdv.load_statistics` to handle reading statistics from TFRecord and
    text files.
*   Added an extra requirement group `mutual-information`. As a result, barebone
    TFDV does not require `scikit-learn` any more.
*   Added an extra requirement group `visualization`. As a result, barebone TFDV
    does not require `ipython` any more.
*   Added an extra requirement group `all` that specifies all the extra
    dependencies TFDV needs. Use `pip install tensorflow-data-validation[all]`
    to pull in those dependencies.
*   Depends on `pyarrow>=0.16,<0.17`.
*   Depends on `apache-beam[gcp]>=2.20,<3`.
*   Depends on `ipython>=7,<8;python_version>="3"'.
*   Depends on `scikit-learn>=0.18,<0.24'.
*   Depends on `tensorflow>=1.15,!=2.0.*,<3`.
*   Depends on `tensorflow-metadata>=0.22.0,<0.23`.
*   Depends on `tensorflow-transform>=0.22,<0.23`.
*   Depends on `tfx-bsl>=0.22,<0.23`.

## Known Issues

*  (Known issue resolution) It is no longer necessary to use Apache Beam 2.17
   when running TFDV on Windows. The current release of Apache Beam will work.

## Breaking Changes

*   `tfdv.GenerateStatistics` now accepts a PCollection of `pa.RecordBatch`
    instead of `pa.Table`.
*   All the TFDV coders now output a PCollection of `pa.RecordBatch` instead of
    a PCollection of `pa.Table`.
*   `tfdv.validate_instances` and
    `tfdv.api.validation_api.IdentifyAnomalousExamples` now takes
    `pa.RecordBatch` as input instead of `pa.Table`.
*   The `StatsGenerator` interface (and all its sub-classes) now takes
    `pa.RecordBatch` as the input data instead of `pa.Table`.
*   Custom slicing functions now accepts a `pa.RecordBatch` instead of
    `pa.Table` as input and should output a tuple `(slice_key, record_batch)`.

## Deprecations

*   Deprecating Py2 support.
