
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

For instructions on using TFDV, see the
[get started guide](https://github.com/tensorflow/data-validation/blob/master/g3doc/get_started.md)
and try out the
[example notebook](https://colab.research.google.com/github/tensorflow/tfx/blob/master/docs/tutorials/data_validation/tfdv_basic.ipynb).
Some of the techniques implemented in TFDV are described in a
[technical paper published in SysML'19](https://www.sysml.cc/doc/2019/167.pdf).

Caution: TFDV may be backwards incompatible before version 1.0.

## Installing from PyPI

The recommended way to install TFDV is using the
[PyPI package](https://pypi.org/project/tensorflow-data-validation/):

```bash
pip install tensorflow-data-validation
```

## Build with Docker

This is the recommended way to build TFDV under Linux, and is continuously
tested at Google.

### 1. Install Docker

Please first install `docker` and `docker-compose` by following the directions:
[docker](https://docs.docker.com/install/);
[docker-compose](https://docs.docker.com/compose/install/).

### 2. Clone the TFDV repository

```shell
git clone https://github.com/tensorflow/data-validation
cd data-validation
```

Note that these instructions will install the latest master branch of TensorFlow
Data Validation. If you want to install a specific branch (such as a release
branch), pass `-b <branchname>` to the `git clone` command.

When building on Python 2, make sure to strip the Python types in the source
code using the following commands:

```shell
pip install strip-hints
python tensorflow_data_validation/tools/strip_type_hints.py tensorflow_data_validation/
```
### 3. Build the pip package

Then, run the following at the project root:

```bash
sudo docker-compose build manylinux2010
sudo docker-compose run -e PYTHON_VERSION=${PYTHON_VERSION} manylinux2010
```
where `PYTHON_VERSION` is one of `{27, 35, 36, 37}`.

A wheel will be produced under `dist/`.

### 4. Install the pip package

```shell
pip install dist/*.whl
```

## Build from source

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

When building on Python 2, make sure to strip the Python types in the source
code using the following commands:

```shell
pip install strip-hints
python tensorflow_data_validation/tools/strip_type_hints.py tensorflow_data_validation/
```

### 3. Build the pip package

TFDV uses Bazel to build the pip package from source. Before invoking the
following commands, make sure the `python` in your `$PATH` is the one of the
target version and has NumPy installed.

```shell
bazel run -c opt --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 tensorflow_data_validation:build_pip_package
```

Note that we are assuming here that dependent packages (e.g. PyArrow) are built
with a GCC older than 5.1 and use the flag `D_GLIBCXX_USE_CXX11_ABI=0` to be
[compatible with the old std::string ABI](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html).

You can find the generated `.whl` file in the `dist` subdirectory.

### 4. Install the pip package

```shell
pip install dist/*.whl
```

## Supported platforms

TFDV is tested on the following 64-bit operating systems:

  * macOS 10.12.6 (Sierra) or later.
  * Ubuntu 16.04 or later.
  * Windows 7 or later.

## Dependencies

TFDV requires TensorFlow but does not depend on the `tensorflow`
[PyPI package](https://pypi.org/project/tensorflow/). See the [TensorFlow install guides](https://www.tensorflow.org/install/)
for instructions on how to get started with TensorFlow.

[Apache Beam](https://beam.apache.org/) is required; it's the way that efficient
distributed computation is supported. By default, Apache Beam runs in local
mode but can also run in distributed mode using
[Google Cloud Dataflow](https://cloud.google.com/dataflow/).
TFDV is designed to be extensible for other Apache Beam runners.

[Apache Arrow](https://arrow.apache.org/) is also required. TFDV uses Arrow to
represent data internally in order to make use of vectorized numpy functions.

## Compatible versions

The following table shows the  package versions that are
compatible with each other. This is determined by our testing framework, but
other *untested* combinations may also work.

tensorflow-data-validation                                                            | tensorflow        | apache-beam[gcp] | pyarrow
------------------------------------------------------------------------------------- | ----------------- | ---------------- | -------
[GitHub master](https://github.com/tensorflow/data-validation/blob/master/RELEASE.md) | nightly (1.x/2.x) | 2.17.0           | 0.15.0
[0.21.1](https://github.com/tensorflow/data-validation/blob/v0.21.1/RELEASE.md)       | 1.15 / 2.1        | 2.17.0           | 0.15.0
[0.21.0](https://github.com/tensorflow/data-validation/blob/v0.21.0/RELEASE.md)       | 1.15 / 2.1        | 2.17.0           | 0.15.0
[0.15.0](https://github.com/tensorflow/data-validation/blob/v0.15.0/RELEASE.md)       | 1.15 / 2.0        | 2.16.0           | 0.14.0
[0.14.1](https://github.com/tensorflow/data-validation/blob/v0.14.1/RELEASE.md)       | 1.14              | 2.14.0           | 0.14.0
[0.14.0](https://github.com/tensorflow/data-validation/blob/v0.14.0/RELEASE.md)       | 1.14              | 2.14.0           | 0.14.0
[0.13.1](https://github.com/tensorflow/data-validation/blob/v0.13.1/RELEASE.md)       | 1.13              | 2.11.0           | n/a
[0.13.0](https://github.com/tensorflow/data-validation/blob/v0.13.0/RELEASE.md)       | 1.13              | 2.11.0           | n/a
[0.12.0](https://github.com/tensorflow/data-validation/blob/v0.12.0/RELEASE.md)       | 1.12              | 2.10.0           | n/a
[0.11.0](https://github.com/tensorflow/data-validation/blob/v0.11.0/RELEASE.md)       | 1.11              | 2.8.0            | n/a
[0.9.0](https://github.com/tensorflow/data-validation/blob/v0.9.0/RELEASE.md)         | 1.9               | 2.6.0            | n/a

## Questions

Please direct any questions about working with TF Data Validation to
[Stack Overflow](https://stackoverflow.com) using the
[tensorflow-data-validation](https://stackoverflow.com/questions/tagged/tensorflow-data-validation)
tag.

## Links

*   [TensorFlow Data Validation Getting Started Guide](https://www.tensorflow.org/tfx/data_validation/get_started)
*   [TensorFlow Data Validation Notebook](https://colab.research.google.com/github/tensorflow/tfx/blob/master/docs/tutorials/data_validation/tfdv_basic.ipynb)
*   [TensorFlow Data Validation API Documentation](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv)
*   [TensorFlow Data Validation Blog Post](https://medium.com/tensorflow/introducing-tensorflow-data-validation-data-understanding-validation-and-monitoring-at-scale-d38e3952c2f0)
*   [TensorFlow Data Validation PyPI](https://pypi.org/project/tensorflow-data-validation/)
*   [TensorFlow Data Validation Paper](https://www.sysml.cc/doc/2019/167.pdf)
*   [TensorFlow Data Validation Slides](https://conf.slac.stanford.edu/xldb2018/sites/xldb2018.conf.slac.stanford.edu/files/Tues_09.45_NeoklisPolyzotis_Data%20Analysis%20and%20Validation%20\(1\).pdf)
