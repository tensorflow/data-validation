<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.validate_examples_in_tfrecord" />
<meta itemprop="path" content="Stable" />
</div>

# tfdv.validate_examples_in_tfrecord

```python
tfdv.validate_examples_in_tfrecord(
    data_location,
    stats_options,
    output_path=None,
    pipeline_options=None
)
```

Validates TFExamples in TFRecord files.

Runs a Beam pipeline to detect anomalies on a per-example basis. If this
function detects anomalous examples, it generates summary statistics regarding
the set of examples that exhibit each anomaly.

This is a convenience function for users with data in TFRecord format. Users
with data in unsupported file/data formats, or users who wish to create their
own Beam pipelines need to use the 'IdentifyAnomalousExamples' PTransform API
directly instead.

#### Args:

*   <b>`data_location`</b>: The location of the input data files.
*   <b>`stats_options`</b>:
    <a href="../tfdv/StatsOptions.md"><code>tfdv.StatsOptions</code></a> for
    generating data statistics. This must contain a schema.
*   <b>`output_path`</b>: The file path to output data statistics result to. If
    None, the function uses a temporary directory. The output will be a TFRecord
    file containing a single data statistics list proto, and can be read with
    the 'load_statistics' function. If you run this function on Google Cloud,
    you must specify an output_path. Specifying None may cause an error.
*   <b>`pipeline_options`</b>: Optional beam pipeline options. This allows users
    to specify various beam pipeline execution parameters like pipeline runner
    (DirectRunner or DataflowRunner), cloud dataflow service project id, etc.
    See https://cloud.google.com/dataflow/pipelines/specifying-exec-params for
    more details.

#### Returns:

A DatasetFeatureStatisticsList proto in which each dataset consists of the set
of examples that exhibit a particular anomaly.

#### Raises:

*   <b>`ValueError`</b>: If the specified stats_options does not include a
    schema.
