<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.generate_statistics_from_dataframe" />
<meta itemprop="path" content="Stable" />
</div>

# tfdv.generate_statistics_from_dataframe

``` python
tfdv.generate_statistics_from_dataframe(
    dataframe,
    stats_options=options.StatsOptions()
)
```

Compute data statistics for the input pandas DataFrame.

This is a utility method for users with in-memory data represented
as a pandas DataFrame.

#### Args:

* <b>`dataframe`</b>: Input pandas DataFrame.
* <b>`stats_options`</b>: Options for generating data statistics.


#### Returns:

A DatasetFeatureStatisticsList proto.