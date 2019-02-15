<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.validate_statistics" />
<meta itemprop="path" content="Stable" />
</div>

# tfdv.validate_statistics

``` python
tfdv.validate_statistics(
    statistics,
    schema,
    environment=None,
    previous_statistics=None,
    serving_statistics=None
)
```

Validates the input statistics against the provided input schema.

This method validates the `statistics` against the `schema`. If an optional
`environment` is specified, the `schema` is filtered using the
`environment` and the `statistics` is validated against the filtered schema.
The optional `previous_statistics` and `serving_statistics` are the statistics
computed over the treatment data for drift- and skew-detection, respectively.

#### Args:

* <b>`statistics`</b>: A DatasetFeatureStatisticsList protocol buffer denoting the
      statistics computed over the current data. Validation is currently only
      supported for lists with a single DatasetFeatureStatistics proto.
* <b>`schema`</b>: A Schema protocol buffer.
* <b>`environment`</b>: An optional string denoting the validation environment.
      Must be one of the default environments specified in the schema.
      By default, validation assumes that all Examples in a pipeline adhere
      to a single schema. In some cases introducing slight schema variations
      is necessary, for instance features used as labels are required during
      training (and should be validated), but are missing during serving.
      Environments can be used to express such requirements. For example,
      assume a feature named 'LABEL' is required for training, but is expected
      to be missing from serving. This can be expressed by defining two
      distinct environments in schema: ["SERVING", "TRAINING"] and
      associating 'LABEL' only with environment "TRAINING".
* <b>`previous_statistics`</b>: An optional DatasetFeatureStatisticsList protocol
      buffer denoting the statistics computed over an earlier data (for
      example, previous day's data). If provided, the `validate_statistics`
      method will detect if there exists drift between current data and
      previous data. Configuration for drift detection can be done by
      specifying a `drift_comparator` in the schema. For now drift detection
      is only supported for categorical features.
* <b>`serving_statistics`</b>: An optional DatasetFeatureStatisticsList protocol
      buffer denoting the statistics computed over the serving data. If
      provided, the `validate_statistics` method will identify if there exists
      distribution skew between current data and serving data. Configuration
      for skew detection can be done by specifying a `skew_comparator` in the
      schema. For now skew detection is only supported for categorical
      features.


#### Returns:

An Anomalies protocol buffer.


#### Raises:

* <b>`TypeError`</b>: If any of the input arguments is not of the expected type.
* <b>`ValueError`</b>: If the input statistics proto does not have only one dataset.