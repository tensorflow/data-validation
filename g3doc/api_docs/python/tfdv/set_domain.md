<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.set_domain" />
<meta itemprop="path" content="Stable" />
</div>

# tfdv.set_domain

```python
tfdv.set_domain(
    schema,
    feature_path,
    domain
)
```

Sets the domain for the input feature in the schema.

If the input feature already has a domain, it is overwritten with the newly
provided input domain. This method cannot be used to add a new global domain.

#### Args:

*   <b>`schema`</b>: A Schema protocol buffer.
*   <b>`feature_path`</b>: The name of the feature whose domain needs to be set.
    If a FeatureName is passed, a one-step FeaturePath will be constructed and
    used. For example, "my_feature" -> types.FeaturePath(["my_feature"])
*   <b>`domain`</b>: A domain protocol buffer (one of IntDomain, FloatDomain,
    StringDomain or BoolDomain) or the name of a global string domain present in
    the input schema. Example: `python >>> from tensorflow_metadata.proto.v0
    import schema_pb2 >>> import tensorflow_data_validation as tfdv >>> schema =
    schema_pb2.Schema() >>> schema.feature.add(name='feature') # Setting a int
    domain. >>> int_domain = schema_pb2.IntDomain(min=3, max=5) >>>
    tfdv.set_domain(schema, "feature", int_domain) # Setting a string
    domain. >>> str_domain = schema_pb2.StringDomain(value=['one', 'two',
    'three']) >>> tfdv.set_domain(schema, "feature", str_domain)`

#### Raises:

* <b>`TypeError`</b>: If the input schema or the domain is not of the expected type.
* <b>`ValueError`</b>: If an invalid global string domain is provided as input.