<!-- See: www.tensorflow.org/tfx/data_validation/ -->

# TensorFlow Data Validation [![PyPI](https://img.shields.io/pypi/pyversions/tensorflow-data-validation.svg?style=plastic)](https://github.com/tensorflow/data-validation)

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

For instructions on using TF Data Validation, see the [get started guide](g3doc/get_started.md).

Caution: `tf.DataValidation` may be backwards incompatible before version 1.0.

## Installing from PyPI

The recommended way to install TensorFlow Data Validation is using the
[PyPI package](https://pypi.org/project/tensorflow-data-validation/):

```bash
pip install tensorflow-data-validation
```

## Installing from source

### 1. Prerequisites

To compile and use TensorFlow Data Validation, you need to set up some prerequisites.

#### Install NumPy

If NumPy is not installed on your system, install it now by following [these
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

### 3. Build and install pip package

TensorFlow Data Validation uses Bazel to build and install the pip package from source:

```shell
bazel run -c opt tensorflow_data_validation:pip_installer
```

Note that the previous command performs a `pip install` in the current python
environment. You can find the installed `.whl` file in the `dist`
subdirectory. It is also possible to pass options to the executed `pip install`
through the environment variable `TFDV_PIP_INSTALL_OPTIONS`.

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

|tensorflow-data-validation |tensorflow    |apache-beam[gcp]|
|---------------------------|--------------|----------------|
|GitHub master              |nightly (1.x) |2.6.0           |
|0.9.0                      |1.9           |2.6.0           |

## Questions

Please direct any questions about working with TF Data Validation to
[Stack Overflow](https://stackoverflow.com) using the
[tensorflow-data-validation](https://stackoverflow.com/questions/tagged/tensorflow-data-validation)
tag.
