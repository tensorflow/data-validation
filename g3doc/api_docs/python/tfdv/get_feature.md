<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.get_feature" />
<meta itemprop="path" content="Stable" />
</div>

# tfdv.get_feature

```python
tfdv.get_feature(
    schema,
    feature_path
)
```

Get a feature from the schema.

#### Args:

*   <b>`schema`</b>: A Schema protocol buffer.
*   <b>`feature_path`</b>: The path of the feature to obtain from the schema. If
    a FeatureName is passed, a one-step FeaturePath will be constructed and
    used. For example, "my_feature" -> types.FeaturePath(["my_feature"])

#### Returns:

A Feature protocol buffer.


#### Raises:

* <b>`TypeError`</b>: If the input schema is not of the expected type.
* <b>`ValueError`</b>: If the input feature is not found in the schema.