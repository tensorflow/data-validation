<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.StatsOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="generators"/>
<meta itemprop="property" content="feature_whitelist"/>
<meta itemprop="property" content="schema"/>
<meta itemprop="property" content="num_top_values"/>
<meta itemprop="property" content="num_rank_histogram_buckets"/>
<meta itemprop="property" content="num_values_histogram_buckets"/>
<meta itemprop="property" content="num_histogram_buckets"/>
<meta itemprop="property" content="num_quantiles_histogram_buckets"/>
<meta itemprop="property" content="epsilon"/>
<meta itemprop="property" content="__new__"/>
</div>

# tfdv.StatsOptions

## Class `StatsOptions`



Options for generating data statistics.

#### Attributes:

* <b>`generators`</b>: An optional list of statistics generators. A statistics
    generator must extend either CombinerStatsGenerator or
    TransformStatsGenerator.
* <b>`feature_whitelist`</b>: An optional list of names of the features to calculate
    statistics for.
* <b>`schema`</b>: An optional tensorflow_metadata Schema proto. Currently we use the
    schema to infer categorical and bytes features.
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
    unequal buckets, but could improve performance, and resource consumption.

<h2 id="__new__"><code>__new__</code></h2>

``` python
@staticmethod
__new__(
    cls,
    generators=None,
    feature_whitelist=None,
    schema=None,
    num_top_values=20,
    num_rank_histogram_buckets=1000,
    num_values_histogram_buckets=10,
    num_histogram_buckets=10,
    num_quantiles_histogram_buckets=10,
    epsilon=0.01
)
```

Create new instance of StatsOptions(generators, feature_whitelist, schema, num_top_values, num_rank_histogram_buckets, num_values_histogram_buckets, num_histogram_buckets, num_quantiles_histogram_buckets, epsilon)



## Properties

<h3 id="generators"><code>generators</code></h3>



<h3 id="feature_whitelist"><code>feature_whitelist</code></h3>



<h3 id="schema"><code>schema</code></h3>



<h3 id="num_top_values"><code>num_top_values</code></h3>



<h3 id="num_rank_histogram_buckets"><code>num_rank_histogram_buckets</code></h3>



<h3 id="num_values_histogram_buckets"><code>num_values_histogram_buckets</code></h3>



<h3 id="num_histogram_buckets"><code>num_histogram_buckets</code></h3>



<h3 id="num_quantiles_histogram_buckets"><code>num_quantiles_histogram_buckets</code></h3>



<h3 id="epsilon"><code>epsilon</code></h3>





