<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.visualize_statistics" />
<meta itemprop="path" content="Stable" />
</div>

# tfdv.visualize_statistics

``` python
tfdv.visualize_statistics(
    lhs_statistics,
    rhs_statistics=None,
    lhs_name='lhs_statistics',
    rhs_name='rhs_statistics'
)
```

Visualize the input statistics using Facets.

#### Args:

* <b>`lhs_statistics`</b>: A DatasetFeatureStatisticsList protocol buffer.
* <b>`rhs_statistics`</b>: An optional DatasetFeatureStatisticsList protocol buffer to
    compare with lhs_statistics.
* <b>`lhs_name`</b>: Name of the lhs_statistics dataset.
* <b>`rhs_name`</b>: Name of the rhs_statistics dataset.


#### Raises:

* <b>`TypeError`</b>: If the input argument is not of the expected type.
* <b>`ValueError`</b>: If the input statistics protos does not have only one dataset.