<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.get_domain" />
<meta itemprop="path" content="Stable" />
</div>

# tfdv.get_domain

``` python
tfdv.get_domain(
    schema,
    feature_name
)
```

Get the domain associated with the input feature from the schema.

#### Args:

* <b>`schema`</b>: A Schema protocol buffer.
* <b>`feature_name`</b>: The name of the feature whose domain needs to be found.


#### Returns:

The domain protocol buffer (one of IntDomain, FloatDomain, StringDomain or
    BoolDomain) associated with the input feature.


#### Raises:

* <b>`TypeError`</b>: If the input schema is not of the expected type.
* <b>`ValueError`</b>: If the input feature is not found in the schema or there is
      no domain associated with the feature.