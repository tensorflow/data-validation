<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.update_schema" />
<meta itemprop="path" content="Stable" />
</div>

# tfdv.update_schema

```python
tfdv.update_schema(
    schema,
    statistics,
    infer_feature_shape=True,
    max_string_domain_size=100
)
```

Updates input schema to conform to the input statistics.

#### Args:

*   <b>`schema`</b>: A Schema protocol buffer.
*   <b>`statistics`</b>: A DatasetFeatureStatisticsList protocol buffer. Schema
    inference is currently only supported for lists with a single
    DatasetFeatureStatistics proto.
*   <b>`infer_feature_shape`</b>: A boolean to indicate if shape of the features
    need to be inferred from the statistics.
*   <b>`max_string_domain_size`</b>: Maximum size of the domain of a string
    feature in order to be interpreted as a categorical feature.

#### Returns:

A Schema protocol buffer.

#### Raises:

*   <b>`TypeError`</b>: If the input argument is not of the expected type.
*   <b>`ValueError`</b>: If the input statistics proto does not have only one
    dataset.
