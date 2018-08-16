# TensorFlow Data Validation

*TensorFlow Data Validation* is a library for exploring and validating
machine learning data. `tf.DataValidation` is designed to be highly scalable
and to work well with TensorFlow and [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx).

TF Data Validation includes:

*    Scalable calculation of summary statistics of training and test data.
*    Integration with a viewer for data distributions and statistics, as well
     as faceted comparison of pairs of features ([Facets](https://github.com/PAIR-code/facets))
*    Automated [data-schema](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto)
     generation to describe expectations about data
     like required values, ranges, and vocabularies
*    A schema viewer to help you inspect the schema.
*    Anomaly detection to identify anomalies, such as missing features,
     out-of-range values, or wrong feature types, to name a few.
*    An anomalies viewer so that you can see what features have anomalies and
     learn more in order to correct them.

See the [examples](https://github.com/tensorflow/data-validation/tree/master/examples)
to learn how to use TF Data Validation and the [quick guide to get started](https://github.com/tensorflow/data-validation/tree/master/g3doc/get_started.md).

Caution: `tf.DataValidation` may be backwards incompatible before version 1.0.

## Installing from PyPI

Note: PyPI package is not available yet.

The `tensorflow-data-validation`
[PyPI package](https://pypi.org/project/tensorflow-data-validation/) is the
recommended way to install `tf.DataValidation`:

```bash
pip install tensorflow-data-validation
```

## Installing from source

### 1. Prerequisites

To compile and use TensorFlow Data Validation, you need to set up some prerequisites.

#### Install NumPy

If NumPy is not installed on your system, install it now following [these
directions](https://www.scipy.org/scipylib/download.html).

#### Install Bazel

If bazel is not installed on your system, install it now by following [these
directions](https://bazel.build/versions/master/docs/install.html).

### 2. Clone the TensorFlow Data Validation repository

```shell
git clone https://github.com/tensorflow/data-validation
cd data-validation
```

Note that these instructions will install the latest master branch of TensorFlow
Data Validation. If you want to install a specific branch (such as a release branch),
pass `-b <branchname>` to the `git clone` command.

### 3. Build

TensorFlow Data Validation uses Bazel to build. Use Bazel commands to build individual
targets or the entire source tree.

To build the Python wrappers for the C++ modules, execute:

```shell
bazel build -c opt tensorflow_data_validation/anomalies:pywrap_tensorflow_data_validation
```

### 4. Copy over generated Python wrappers

```shell
cp bazel-bin/tensorflow_data_validation/anomalies/_pywrap_tensorflow_data_validation.so tensorflow_data_validation/anomalies/
cp bazel-bin/tensorflow_data_validation/anomalies/pywrap_tensorflow_data_validation.py tensorflow_data_validation/anomalies/
```

### 5. Build the pip package

Run the following command to create a `.whl` file within `dist` directory.

```shell
python setup.py bdist_wheel
```

### 6. Install the pip package

Invoke `pip install` to install that pip package. The filename of the `.whl` file depends on your platform.

```shell
pip install dist/*.whl
```

## Dependencies

`tf.DataValidation` requires TensorFlow but does not depend on the `tensorflow`
[PyPI package](https://pypi.org/project/tensorflow/). See the[TensorFlow install guides](https://www.tensorflow.org/install/)
for instructions on how to get started with TensorFlow.

[Apache Beam](https://beam.apache.org/) is required; it's the way that efficient
distributed computation is supported. By default, Apache Beam runs in local
mode but can also run in distributed mode using
[Google Cloud Dataflow](https://cloud.google.com/dataflow/).
`tf.DataValidation` is designed to be extensible for other Apache Beam runners.

## Compatible versions

The following table shows the `tf.DataValidation` package versions that are
compatible with each other. This is determined by our testing framework, but
other *untested* combinations may also work.

|tensorflow-data-validation                                                            |tensorflow    |apache-beam[gcp]|
|--------------------------------------------------------------------------------------|--------------|----------------|
|GitHub master |nightly (1.x) |2.5.0           |
