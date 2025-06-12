# Custom Data Validation

<!--*
freshness: { owner: 'kuochuntsai' reviewed: '2022-11-29' }
*-->

TFDV supports custom data validation using SQL. You can run custom data
validation using
[validate_statistics](https://github.com/tensorflow/data-validation/blob/master/tensorflow_data_validation/api/validation_api.py;l=236;rcl=488721853)
or
[custom_validate_statistics](https://github.com/tensorflow/data-validation/blob/master/tensorflow_data_validation/api/validation_api.py;l=535;rcl=488721853).
Use `validate_statistics` to run standard, schema-based data validation along
with custom validation. Use `custom_validate_statistics` to run only custom
validation.

## Configuring Custom Data Validation

Use the
[CustomValidationConfig](https://github.com/tensorflow/data-validation/blob/master/tensorflow_data_validation/anomalies/proto/custom_validation_config.proto)
to define custom validations to run. For each validation, provide an
SQL expression, which returns a boolean value. Each SQL expression is run
against the summary statistics for the specified feature. If the expression
returns false, TFDV generates a custom anomaly using the provided severity and
anomaly description.

You may configure custom validations that run against individual features or
feature pairs. For each feature, specify both the dataset (i.e., slice) and the
feature path to use, though you may leave the dataset name blank if you want to
validate the default slice (i.e., all examples). For single feature validations,
the feature statistics are bound to `feature`. For feature pair validations, the
test feature statistics are bound to `feature_test` and the base feature
statistics are bound to `feature_base`. See the section below for example
queries.

If a custom validation triggers an anomaly, TFDV will return an Anomalies proto
with the reason(s) for the anomaly. Each reason will have a short description,
which is user configured, and a description with the query that caused the
anomaly, the dataset names on which the query was run, and the base feature path
(if running a feature-pair validation). See the section below for example
results of custom validation.

See the
[documentation](https://github.com/tensorflow/data-validation/blob/master/tensorflow_data_validation/anomalies/proto/custom_validation_config.proto)
in the `CustomValidationConfig` proto for example
configurations.
