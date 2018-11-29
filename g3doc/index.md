
{% setvar github_path %}tensorflow/data-validation{% endsetvar %}
{% include "_templates/github-bug.html" %}

# TensorFlow Data Validation

*TensorFlow Data Validation* (TFDV) is a library for exploring and validating
machine learning data. It is designed to be highly scalable
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

For instructions on using TFDV, see the [get started guide](get_started.md)
and try out the [example notebook](https://colab.research.google.com/github/tensorflow/data-validation/blob/master/examples/chicago_taxi/chicago_taxi_tfdv.ipynb).

Caution: TFDV may be backwards incompatible before version 1.0.

## Installing from PyPI

The recommended way to install TFDV is using the
[PyPI package](https://pypi.org/project/tensorflow-data-validation/):

```bash
pip install tensorflow-data-validation
```

## Installing from source

### 1. Prerequisites

To compile and use TFDV, you need to set up some prerequisites.

#### Install NumPy

If NumPy is not installed on your system, install it now by following [these
directions](https://www.scipy.org/scipylib/download.html).

#### Install Bazel

If Bazel is not installed on your system, install it now by following [these
directions](https://bazel.build/versions/master/docs/install.html).

### 2. Clone the TFDV repository

```shell
git clone https://github.com/tensorflow/data-validation
cd data-validation
```

Note that these instructions will install the latest master branch of TensorFlow
Data Validation. If you want to install a specific branch (such as a release branch),
pass `-b <branchname>` to the `git clone` command.

### 3. Build the pip package

TFDV uses Bazel to build the pip package from source:

```shell
bazel run -c opt tensorflow_data_validation:build_pip_package
```

You can find the generated `.whl` file in the `dist` subdirectory.

### 4. Install the pip package

```shell
pip install dist/*.whl
```

## Supported platforms

Note: TFDV currently requires Python 2.7. Support for Python 3 is coming
very soon (tracked [here](https://github.com/tensorflow/data-validation/issues/10)).

TFDV is built and tested on the following 64-bit operating systems:

  * macOS 10.12.6 (Sierra) or later.
  * Ubuntu 14.04 or later.

## Dependencies

TFDV requires TensorFlow but does not depend on the `tensorflow`
[PyPI package](https://pypi.org/project/tensorflow/). See the [TensorFlow install guides](https://www.tensorflow.org/install/)
for instructions on how to get started with TensorFlow.

[Apache Beam](https://beam.apache.org/) is required; it's the way that efficient
distributed computation is supported. By default, Apache Beam runs in local
mode but can also run in distributed mode using
[Google Cloud Dataflow](https://cloud.google.com/dataflow/).
TFDV is designed to be extensible for other Apache Beam runners.

## Compatible versions

The following table shows the  package versions that are
compatible with each other. This is determined by our testing framework, but
other *untested* combinations may also work.

|tensorflow-data-validation |tensorflow    |apache-beam[gcp]|
|---------------------------|--------------|----------------|
|GitHub master              |nightly (1.x) |2.8.0           |
|0.11.0                     |1.11          |2.8.0           |
|0.9.0                      |1.9           |2.6.0           |

## Questions

Please direct any questions about working with TF Data Validation to
[Stack Overflow](https://stackoverflow.com) using the
[tensorflow-data-validation](https://stackoverflow.com/questions/tagged/tensorflow-data-validation)
tag.
