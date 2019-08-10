<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.StatsOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="desired_batch_size"/>
<meta itemprop="property" content="feature_whitelist"/>
<meta itemprop="property" content="generators"/>
<meta itemprop="property" content="num_histogram_buckets"/>
<meta itemprop="property" content="num_quantiles_histogram_buckets"/>
<meta itemprop="property" content="num_values_histogram_buckets"/>
<meta itemprop="property" content="sample_count"/>
<meta itemprop="property" content="sample_rate"/>
<meta itemprop="property" content="schema"/>
<meta itemprop="property" content="semantic_domain_stats_sample_rate"/>
<meta itemprop="property" content="slice_functions"/>
<meta itemprop="property" content="__init__"/>
</div>

# tfdv.StatsOptions

## Class `StatsOptions`



Options for generating statistics.

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(
    generators=None,
    feature_whitelist=None,
    schema=None,
    weight_feature=None,
    slice_functions=None,
    sample_count=None,
    sample_rate=None,
    num_top_values=20,
    frequency_threshold=1,
    weighted_frequency_threshold=1.0,
    num_rank_histogram_buckets=1000,
    num_values_histogram_buckets=10,
    num_histogram_buckets=10,
    num_quantiles_histogram_buckets=10,
    epsilon=0.01,
    infer_type_from_schema=False,
    desired_batch_size=None,
    enable_semantic_domain_stats=False,
    semantic_domain_stats_sample_rate=None
)
```

Initializes statistics options.

#### Args:

*   <b>`generators`</b>: An optional list of statistics generators. A statistics
    generator must extend either CombinerStatsGenerator or
    TransformStatsGenerator.
*   <b>`feature_whitelist`</b>: An optional list of names of the features to
    calculate statistics for.
*   <b>`schema`</b>: An optional tensorflow_metadata Schema proto. Currently we
    use the schema to infer categorical and bytes features.
*   <b>`weight_feature`</b>: An optional feature name whose numeric value
    represents the weight of an example.
*   <b>`slice_functions`</b>: An optional list of functions that generate slice
    keys for each example. Each slice function should take an example dict as
    input and return a list of zero or more slice keys.
*   <b>`sample_count`</b>: An optional number of examples to include in the
    sample. If specified, statistics is computed over the sample. Only one of
    sample_count or sample_rate can be specified. Note that since TFDV batches
    input examples, the sample count is only a desired count and we may include
    more examples in certain cases.
*   <b>`sample_rate`</b>: An optional sampling rate. If specified, statistics is
    computed over the sample. Only one of sample_count or sample_rate can be
    specified.
*   <b>`num_top_values`</b>: An optional number of most frequent feature values
    to keep for string features.
*   <b>`frequency_threshold`</b>: An optional minimum number of examples the
    most frequent values must be present in.
*   <b>`weighted_frequency_threshold`</b>: An optional minimum weighted number
    of examples the most frequent weighted values must be present in. This
    option is only relevant when a weight_feature is specified.
*   <b>`num_rank_histogram_buckets`</b>: An optional number of buckets in the
    rank histogram for string features.
*   <b>`num_values_histogram_buckets`</b>: An optional number of buckets in a
    quantiles histogram for the number of values per Feature, which is stored in
    CommonStatistics.num_values_histogram.
*   <b>`num_histogram_buckets`</b>: An optional number of buckets in a standard
    NumericStatistics.histogram with equal-width buckets.
*   <b>`num_quantiles_histogram_buckets`</b>: An optional number of buckets in a
    quantiles NumericStatistics.histogram.
*   <b>`epsilon`</b>: An optional error tolerance for the computation of
    quantiles, typically a small fraction close to zero (e.g. 0.01). Higher
    values of epsilon increase the quantile approximation, and hence result in
    more unequal buckets, but could improve performance, and resource
    consumption.
*   <b>`infer_type_from_schema`</b>: A boolean to indicate whether the feature
    types should be inferred from the schema. If set to True, an input schema
    must be provided. This flag is used only when generating statistics on CSV
    data.
*   <b>`desired_batch_size`</b>: An optional number of examples to include in
    each batch that is passed to the statistics generators.
*   <b>`enable_semantic_domain_stats`</b>: If True statistics for semantic
    domains are generated (e.g: image, text domains).
*   <b>`semantic_domain_stats_sample_rate`</b>: An optional sampling rate for
    semantic domain statistics. If specified, semantic domain statistics is
    computed over a sample.

## Properties

<h3 id="desired_batch_size"><code>desired_batch_size</code></h3>



<h3 id="feature_whitelist"><code>feature_whitelist</code></h3>



<h3 id="generators"><code>generators</code></h3>



<h3 id="num_histogram_buckets"><code>num_histogram_buckets</code></h3>



<h3 id="num_quantiles_histogram_buckets"><code>num_quantiles_histogram_buckets</code></h3>



<h3 id="num_values_histogram_buckets"><code>num_values_histogram_buckets</code></h3>



<h3 id="sample_count"><code>sample_count</code></h3>



<h3 id="sample_rate"><code>sample_rate</code></h3>



<h3 id="schema"><code>schema</code></h3>

<h3 id="semantic_domain_stats_sample_rate"><code>semantic_domain_stats_sample_rate</code></h3>

<h3 id="slice_functions"><code>slice_functions</code></h3>





