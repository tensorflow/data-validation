<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.generate_statistics_from_csv" />
<meta itemprop="path" content="Stable" />
</div>

# tfdv.generate_statistics_from_csv

```python
tfdv.generate_statistics_from_csv(
    data_location,
    column_names=None,
    delimiter=',',
    output_path=None,
    stats_options=options.StatsOptions(),
    pipeline_options=None,
    compression_type=CompressionTypes.AUTO
)
```

Compute data statistics from CSV files.

Runs a Beam pipeline to compute the data statistics and return the result
data statistics proto.

This is a convenience method for users with data in CSV format.
Users with data in unsupported file/data formats, or users who wish
to create their own Beam pipelines need to use the 'GenerateStatistics'
PTransform API directly instead.

#### Args:

*   <b>`data_location`</b>: The location of the input data files.
*   <b>`column_names`</b>: A list of column names to be treated as the CSV
    header. Order must match the order in the input CSV files. If this argument
    is not specified, we assume the first line in the input CSV files as the
    header. Note that this option is valid only for 'csv' input file format.
*   <b>`delimiter`</b>: A one-character string used to separate fields in a CSV
    file.
*   <b>`output_path`</b>: The file path to output data statistics result to. If
    None, we use a temporary directory. It will be a TFRecord file containing a
    single data statistics proto, and can be read with the 'load_statistics'
    API. If you run this function on Google Cloud, you must specify an
    output_path. Specifying None may cause an error.
*   <b>`stats_options`</b>:
    <a href="../tfdv/StatsOptions.md"><code>tfdv.StatsOptions</code></a> for
    generating data statistics.
*   <b>`pipeline_options`</b>: Optional beam pipeline options. This allows users
    to specify various beam pipeline execution parameters like pipeline runner
    (DirectRunner or DataflowRunner), cloud dataflow service project id, etc.
    See https://cloud.google.com/dataflow/pipelines/specifying-exec-params for
    more details.
*   <b>`compression_type`</b>: Used to handle compressed input files. Default
    value is CompressionTypes.AUTO, in which case the file_path's extension will
    be used to detect the compression.

#### Returns:

A DatasetFeatureStatisticsList proto.