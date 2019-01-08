# Current version (not yet released; still in development)

## Major Features and Improvements

* Add a `validate_instance` function, which checks a single example for anomalies.

## Bug Fixes and Other Changes

* Use constant '__BYTES_VALUE__' in the statistics proto to represent a bytes value which cannot be decoded as a utf-8 string.

## Breaking changes

## Deprecations

# Release 0.11.0

## Major Features and Improvements

* Add option to infer feature types from schema when generating statistics over CSV data.
* Add utility method `set_domain` to set the domain of a feature in the schema.
* Add option to compute weighted statistics by providing a weight feature.
* Add a PTransform for decoding TF examples.
* Add utility methods `write_schema_text` and `load_schema_text` to write and load the schema protocol buffer.
* Add option to compute statistics over a sample.
* Optimize performance of statistics computation (~2x improvement on benchmark datasets).
* Added Getting Started Example Colab and removed previous Jupyter notebook

## Bug Fixes and Other Changes

* Depends on `apache-beam[gcp]>=2.8,<3`.
* Depends on `tensorflow-transform>=0.11,<0.12`.
* Depends on `tensorflow-metadata>=0.9,<0.10`.
* Fix bug in clearing oneof domain\_info field in Feature proto.
* Fix overflow error for large integers by casting them to STRING type.
* Added API docs.

## Breaking changes

* Requires pre-installed `tensorflow>=1.11,<2`.
* Make tf.Example decoder to represent a feature with no value list as a
  missing value (None).
* Make StatsOptions as a class.

## Deprecations

# Release 0.9.0

* Initial release of TensorFlow Data Validation.
