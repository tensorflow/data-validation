<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.infer_schema" />
<meta itemprop="path" content="Stable" />
</div>

# tfdv.infer_schema

``` python
tfdv.infer_schema(
    statistics,
    infer_feature_shape=True,
    max_string_domain_size=100
)
```

Infers schema from the input statistics.

#### Args:

* <b>`statistics`</b>: A DatasetFeatureStatisticsList protocol buffer. Schema
      inference is currently only supported for lists with a single
      DatasetFeatureStatistics proto.
* <b>`infer_feature_shape`</b>: A boolean to indicate if shape of the features need
      to be inferred from the statistics.
* <b>`max_string_domain_size`</b>: Maximum size of the domain of a string feature in
      order to be interpreted as a categorical feature.


#### Returns:

A Schema protocol buffer.


#### Raises:

* <b>`TypeError`</b>: If the input argument is not of the expected type.
* <b>`ValueError`</b>: If the input statistics proto does not have only one dataset.