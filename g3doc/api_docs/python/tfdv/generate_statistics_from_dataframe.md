<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.generate_statistics_from_dataframe" />
<meta itemprop="path" content="Stable" />
</div>

# tfdv.generate_statistics_from_dataframe

``` python
tfdv.generate_statistics_from_dataframe(
    dataframe,
    stats_options=options.StatsOptions(),
    n_jobs=1
)
```

Compute data statistics for the input pandas DataFrame.

This is a utility method for users with in-memory data represented
as a pandas DataFrame.

#### Args:

* <b>`dataframe`</b>: Input pandas DataFrame.
* <b>`stats_options`</b>: <a href="../tfdv/StatsOptions.md"><code>tfdv.StatsOptions</code></a> for generating data statistics.
* <b>`n_jobs`</b>: Number of processes to run (defaults to 1). If -1 is provided,
    uses the same number of processes as the number of CPU cores.


#### Returns:

A DatasetFeatureStatisticsList proto.