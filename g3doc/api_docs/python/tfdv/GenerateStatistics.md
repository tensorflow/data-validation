<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.GenerateStatistics" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="expand"/>
</div>

# tfdv.GenerateStatistics

## Class `GenerateStatistics`



API for generating data statistics.

Example:

```python
  with beam.Pipeline(runner=...) as p:
    _ = (p
         | 'ReadData' >> beam.io.ReadFromTFRecord(data_location)
         | 'DecodeData' >> beam.Map(TFExampleDecoder().decode)
         | 'GenerateStatistics' >> GenerateStatistics()
         | 'WriteStatsOutput' >> beam.io.WriteToTFRecord(
             output_path, shard_name_template='',
             coder=beam.coders.ProtoCoder(
                 statistics_pb2.DatasetFeatureStatisticsList)))
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(options=stats_options.StatsOptions())
```

Initializes the transform.

#### Args:

* <b>`options`</b>: <a href="../tfdv/StatsOptions.md"><code>tfdv.StatsOptions</code></a> for generating data statistics.


#### Raises:

* <b>`TypeError`</b>: If options is not of the expected type.



## Methods

<h3 id="expand"><code>expand</code></h3>

``` python
expand(dataset)
```





