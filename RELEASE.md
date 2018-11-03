# Current version (not yet released; still in development)

## Major Features and Improvements

* Add a PTransform for decoding TF examples.
* Add utility methods to write and load the schema protocol buffer.
* Add option to compute statistics over a sample.
* Add support for computing weighted common statistics.

## Bug Fixes and Other Changes

* Fix bug in clearing oneof domain\_info field in Feature proto.
* Fix overflow error for large integers by casting them to STRING type.
* Added API docs.

## Breaking changes

* Make tf.Example decoder to represent a feature with no value list as a
  missing value (None).
* Make StatsOptions as a class.

## Deprecations

# Release 0.9.0

* Initial release of TensorFlow Data Validation.
