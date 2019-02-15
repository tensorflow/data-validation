<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.TransformStatsGenerator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="ptransform"/>
<meta itemprop="property" content="schema"/>
<meta itemprop="property" content="__init__"/>
</div>

# tfdv.TransformStatsGenerator

## Class `TransformStatsGenerator`



Generate statistics using a Beam PTransform.

Note that the input PTransform must take a PCollection of sliced
examples (tuple of (slice_key, example)) as input and output a
PCollection of sliced protos
(tuple of (slice_key, DatasetFeatureStatistics proto)).

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    name,
    ptransform,
    schema=None
)
```

Initializes a statistics generator.

#### Args:

* <b>`name`</b>: A unique name associated with the statistics generator.
* <b>`schema`</b>: An optional schema for the dataset.



## Properties

<h3 id="name"><code>name</code></h3>



<h3 id="ptransform"><code>ptransform</code></h3>



<h3 id="schema"><code>schema</code></h3>





