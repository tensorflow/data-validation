<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.validate_instance" />
<meta itemprop="path" content="Stable" />
</div>

# tfdv.validate_instance

``` python
tfdv.validate_instance(
    instance,
    options,
    environment=None
)
```

Validates a single example against the schema provided in `options`.

If an optional `environment` is specified, the schema is filtered using the
`environment` and the `instance` is validated against the filtered schema.

#### Args:

* <b>`instance`</b>: A single example in the form of a dict mapping a feature name to a
    numpy array.
* <b>`options`</b>: <a href="../tfdv/StatsOptions.md"><code>tfdv.StatsOptions</code></a> for generating data statistics. This must
    contain a schema.
* <b>`environment`</b>: An optional string denoting the validation environment. Must be
    one of the default environments specified in the schema. In some cases
    introducing slight schema variations is necessary, for instance features
    used as labels are required during training (and should be validated), but
    are missing during serving. Environments can be used to express such
    requirements. For example, assume a feature named 'LABEL' is required for
    training, but is expected to be missing from serving. This can be
    expressed by defining two distinct environments in the schema: ["SERVING",
    "TRAINING"] and associating 'LABEL' only with environment "TRAINING".


#### Returns:

An Anomalies protocol buffer.


#### Raises:

* <b>`ValueError`</b>: If `options` is not a StatsOptions object.
* <b>`ValueError`</b>: If `options` does not contain a schema.