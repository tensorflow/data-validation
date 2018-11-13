<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.StatsOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tfdv.StatsOptions

## Class `StatsOptions`



Options for generating statistics.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    generators=None,
    feature_whitelist=None,
    schema=None,
    weight_feature=None,
    sample_count=None,
    sample_rate=None,
    num_top_values=20,
    num_rank_histogram_buckets=1000,
    num_values_histogram_buckets=10,
    num_histogram_buckets=10,
    num_quantiles_histogram_buckets=10,
    epsilon=0.01,
    infer_type_from_schema=False
)
```

Initializes statistics options.

#### Args:

* <b>`generators`</b>: An optional list of statistics generators. A statistics
    generator must extend either CombinerStatsGenerator or
    TransformStatsGenerator.
* <b>`feature_whitelist`</b>: An optional list of names of the features to calculate
    statistics for.
* <b>`schema`</b>: An optional tensorflow_metadata Schema proto. Currently we use the
    schema to infer categorical and bytes features.
* <b>`weight_feature`</b>: An optional feature name whose numeric value represents
      the weight of an example.
* <b>`sample_count`</b>: An optional number of examples to include in the sample. If
    specified, statistics is computed over the sample. Only one of
    sample_count or sample_rate can be specified.
* <b>`sample_rate`</b>: An optional sampling rate. If specified, statistics is
    computed over the sample. Only one of sample_count or sample_rate can
    be specified.
* <b>`num_top_values`</b>: An optional number of most frequent feature values to keep
    for string features.
* <b>`num_rank_histogram_buckets`</b>: An optional number of buckets in the rank
    histogram for string features.
* <b>`num_values_histogram_buckets`</b>: An optional number of buckets in a quantiles
    histogram for the number of values per Feature, which is stored in
    CommonStatistics.num_values_histogram.
* <b>`num_histogram_buckets`</b>: An optional number of buckets in a standard
    NumericStatistics.histogram with equal-width buckets.
* <b>`num_quantiles_histogram_buckets`</b>: An optional number of buckets in a
    quantiles NumericStatistics.histogram.
* <b>`epsilon`</b>: An optional error tolerance for the computation of quantiles,
    typically a small fraction close to zero (e.g. 0.01). Higher values of
    epsilon increase the quantile approximation, and hence result in more
    unequal buckets, but could improve performance, and resource
    consumption.
* <b>`infer_type_from_schema`</b>: A boolean to indicate whether the feature types
      should be inferred from the schema. If set to True, an input schema
      must be provided. This flag is used only when generating statistics
      on CSV data.



