<!-- See: www.tensorflow.org/tfx/data_validation/ -->

# TensorFlow Data Validation

[![Python](https://img.shields.io/pypi/pyversions/tensorflow-data-validation.svg?style=plastic)](https://github.com/tensorflow/data-validation)
[![PyPI](https://badge.fury.io/py/tensorflow-data-validation.svg)](https://badge.fury.io/py/tensorflow-data-validation)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv)

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
*    Anomaly detection to identify [anomalies](https://github.com/tensorflow/data-validation/blob/master/g3doc/anomalies.md),
     such as missing features,
     out-of-range values, or wrong feature types, to name a few.
*    An anomalies viewer so that you can see what features have anomalies and
     learn more in order to correct them.

For instructions on using TFDV, see the [get started guide](https://github.com/tensorflow/data-validation/blob/master/g3doc/get_started.md)
and try out the [example notebook](https://colab.research.google.com/github/tensorflow/tfx/blob/master/docs/tutorials/data_validation/tfdv_basic.ipynb).
Some of the techniques implemented in TFDV are described in a
[technical paper published in SysML'19](https://mlsys.org/Conferences/2019/doc/2019/167.pdf).

## Installing from PyPI

The recommended way to install TFDV is using the
[PyPI package](https://pypi.org/project/tensorflow-data-validation/):

```bash
pip install tensorflow-data-validation
```
### Nightly Packages

TFDV also hosts nightly packages at https://pypi-nightly.tensorflow.org on
Google Cloud. To install the latest nightly package, please use the following
command:

```bash
export TFX_DEPENDENCY_SELECTOR=NIGHTLY
pip install --extra-index-url https://pypi-nightly.tensorflow.org/simple tensorflow-data-validation
```

This will install the nightly packages for the major dependencies of TFDV such
as TFX Basic Shared Libraries (TFX-BSL) and TensorFlow Metadata (TFMD).

Sometimes TFDV uses those dependencies' most recent changes, which are not yet
released. Because of this, it is safer to use nightly versions of those
dependent libraries when using nightly TFDV. Export the
`TFX_DEPENDENCY_SELECTOR` environment variable to do so.

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

### 3. Build the pip package

Then, run the following at the project root:

```bash
sudo docker-compose build manylinux2010
sudo docker-compose run -e PYTHON_VERSION=${PYTHON_VERSION} manylinux2010
```
where `PYTHON_VERSION` is one of `{37, 38, 39}`.

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
Data Validation. If you want to install a specific branch (such as a release
branch), pass `-b <branchname>` to the `git clone` command.

### 3. Build the pip package

`TFDV` wheel is Python version dependent -- to build the pip package that
works for a specific Python version, use that Python binary to run:

```shell
python setup.py bdist_wheel
```

You can find the generated `.whl` file in the `dist` subdirectory.

### 4. Install the pip package

```shell
pip install dist/*.whl
```

## Supported platforms

TFDV is tested on the following 64-bit operating systems:

  * macOS 10.14.6 (Mojave) or later.
  * Ubuntu 16.04 or later.
  * Windows 7 or later.

## Notable Dependencies

TensorFlow is required.

[Apache Beam](https://beam.apache.org/) is required; it's the way that efficient
distributed computation is supported. By default, Apache Beam runs in local
mode but can also run in distributed mode using
[Google Cloud Dataflow](https://cloud.google.com/dataflow/) and other Apache
Beam
[runners](https://beam.apache.org/documentation/runners/capability-matrix/).

[Apache Arrow](https://arrow.apache.org/) is also required. TFDV uses Arrow to
represent data internally in order to make use of vectorized numpy functions.

## Compatible versions

The following table shows the  package versions that are
compatible with each other. This is determined by our testing framework, but
other *untested* combinations may also work.

tensorflow-data-validation                                                            | apache-beam[gcp] | pyarrow | tensorflow        | tensorflow-metadata | tensorflow-transform | tfx-bsl
------------------------------------------------------------------------------------- | ---------------- | ------- | ----------------- | ------------------- | -------------------- | -------
[GitHub master](https://github.com/tensorflow/data-validation/blob/master/RELEASE.md) | 2.40.0           | 6.0.0   | nightly (1.x/2.x) | 1.10.0              | n/a                  | 1.10.1
[1.10.0](https://github.com/tensorflow/data-validation/blob/v1.10.0/RELEASE.md)       | 2.40.0           | 6.0.0   | 1.15 / 2.9        | 1.10.0              | n/a                  | 1.10.1
[1.9.0](https://github.com/tensorflow/data-validation/blob/v1.9.0/RELEASE.md)         | 2.38.0           | 5.0.0   | 1.15 / 2.9        | 1.9.0               | n/a                  | 1.9.0
[1.8.0](https://github.com/tensorflow/data-validation/blob/v1.8.0/RELEASE.md)         | 2.38.0           | 5.0.0   | 1.15 / 2.8        | 1.8.0               | n/a                  | 1.8.0
[1.7.0](https://github.com/tensorflow/data-validation/blob/v1.7.0/RELEASE.md)         | 2.36.0           | 5.0.0   | 1.15 / 2.8        | 1.7.0               | n/a                  | 1.7.0
[1.6.0](https://github.com/tensorflow/data-validation/blob/v1.6.0/RELEASE.md)         | 2.35.0           | 5.0.0   | 1.15 / 2.7        | 1.6.0               | n/a                  | 1.6.0
[1.5.0](https://github.com/tensorflow/data-validation/blob/v1.5.0/RELEASE.md)         | 2.34.0           | 5.0.0   | 1.15 / 2.7        | 1.5.0               | n/a                  | 1.5.0
[1.4.0](https://github.com/tensorflow/data-validation/blob/v1.4.0/RELEASE.md)         | 2.32.0           | 4.0.1   | 1.15 / 2.6        | 1.4.0               | n/a                  | 1.4.0
[1.3.0](https://github.com/tensorflow/data-validation/blob/v1.3.0/RELEASE.md)         | 2.32.0           | 2.0.0   | 1.15 / 2.6        | 1.2.0               | n/a                  | 1.3.0
[1.2.0](https://github.com/tensorflow/data-validation/blob/v1.2.0/RELEASE.md)         | 2.31.0           | 2.0.0   | 1.15 / 2.5        | 1.2.0               | n/a                  | 1.2.0
[1.1.1](https://github.com/tensorflow/data-validation/blob/v1.1.1/RELEASE.md)         | 2.29.0           | 2.0.0   | 1.15 / 2.5        | 1.1.0               | n/a                  | 1.1.1
[1.1.0](https://github.com/tensorflow/data-validation/blob/v1.1.0/RELEASE.md)         | 2.29.0           | 2.0.0   | 1.15 / 2.5        | 1.1.0               | n/a                  | 1.1.0
[1.0.0](https://github.com/tensorflow/data-validation/blob/v1.0.0/RELEASE.md)         | 2.29.0           | 2.0.0   | 1.15 / 2.5        | 1.0.0               | n/a                  | 1.0.0
[0.30.0](https://github.com/tensorflow/data-validation/blob/v0.30.0/RELEASE.md)       | 2.28.0           | 2.0.0   | 1.15 / 2.4        | 0.30.0              | n/a                  | 0.30.0
[0.29.0](https://github.com/tensorflow/data-validation/blob/v0.29.0/RELEASE.md)       | 2.28.0           | 2.0.0   | 1.15 / 2.4        | 0.29.0              | n/a                  | 0.29.0
[0.28.0](https://github.com/tensorflow/data-validation/blob/v0.28.0/RELEASE.md)       | 2.28.0           | 2.0.0   | 1.15 / 2.4        | 0.28.0              | n/a                  | 0.28.1
[0.27.0](https://github.com/tensorflow/data-validation/blob/v0.27.0/RELEASE.md)       | 2.27.0           | 2.0.0   | 1.15 / 2.4        | 0.27.0              | n/a                  | 0.27.0
[0.26.1](https://github.com/tensorflow/data-validation/blob/v0.26.1/RELEASE.md)       | 2.28.0           | 0.17.0  | 1.15 / 2.3        | 0.26.0              | 0.26.0               | 0.26.0
[0.26.0](https://github.com/tensorflow/data-validation/blob/v0.26.0/RELEASE.md)       | 2.25.0           | 0.17.0  | 1.15 / 2.3        | 0.26.0              | 0.26.0               | 0.26.0
[0.25.0](https://github.com/tensorflow/data-validation/blob/v0.25.0/RELEASE.md)       | 2.25.0           | 0.17.0  | 1.15 / 2.3        | 0.25.0              | 0.25.0               | 0.25.0
[0.24.1](https://github.com/tensorflow/data-validation/blob/v0.24.1/RELEASE.md)       | 2.24.0           | 0.17.0  | 1.15 / 2.3        | 0.24.0              | 0.24.1               | 0.24.1
[0.24.0](https://github.com/tensorflow/data-validation/blob/v0.24.0/RELEASE.md)       | 2.23.0           | 0.17.0  | 1.15 / 2.3        | 0.24.0              | 0.24.0               | 0.24.0
[0.23.1](https://github.com/tensorflow/data-validation/blob/v0.23.1/RELEASE.md)       | 2.24.0           | 0.17.0  | 1.15 / 2.3        | 0.23.0              | 0.23.0               | 0.23.0
[0.23.0](https://github.com/tensorflow/data-validation/blob/v0.23.0/RELEASE.md)       | 2.23.0           | 0.17.0  | 1.15 / 2.3        | 0.23.0              | 0.23.0               | 0.23.0
[0.22.2](https://github.com/tensorflow/data-validation/blob/v0.22.2/RELEASE.md)       | 2.20.0           | 0.16.0  | 1.15 / 2.2        | 0.22.0              | 0.22.0               | 0.22.1
[0.22.1](https://github.com/tensorflow/data-validation/blob/v0.22.1/RELEASE.md)       | 2.20.0           | 0.16.0  | 1.15 / 2.2        | 0.22.0              | 0.22.0               | 0.22.1
[0.22.0](https://github.com/tensorflow/data-validation/blob/v0.22.0/RELEASE.md)       | 2.20.0           | 0.16.0  | 1.15 / 2.2        | 0.22.0              | 0.22.0               | 0.22.0
[0.21.5](https://github.com/tensorflow/data-validation/blob/v0.21.5/RELEASE.md)       | 2.17.0           | 0.15.0  | 1.15 / 2.1        | 0.21.0              | 0.21.1               | 0.21.3
[0.21.4](https://github.com/tensorflow/data-validation/blob/v0.21.4/RELEASE.md)       | 2.17.0           | 0.15.0  | 1.15 / 2.1        | 0.21.0              | 0.21.1               | 0.21.3
[0.21.2](https://github.com/tensorflow/data-validation/blob/v0.21.2/RELEASE.md)       | 2.17.0           | 0.15.0  | 1.15 / 2.1        | 0.21.0              | 0.21.0               | 0.21.0
[0.21.1](https://github.com/tensorflow/data-validation/blob/v0.21.1/RELEASE.md)       | 2.17.0           | 0.15.0  | 1.15 / 2.1        | 0.21.0              | 0.21.0               | 0.21.0
[0.21.0](https://github.com/tensorflow/data-validation/blob/v0.21.0/RELEASE.md)       | 2.17.0           | 0.15.0  | 1.15 / 2.1        | 0.21.0              | 0.21.0               | 0.21.0
[0.15.0](https://github.com/tensorflow/data-validation/blob/v0.15.0/RELEASE.md)       | 2.16.0           | 0.14.0  | 1.15 / 2.0        | 0.15.0              | 0.15.0               | 0.15.0
[0.14.1](https://github.com/tensorflow/data-validation/blob/v0.14.1/RELEASE.md)       | 2.14.0           | 0.14.0  | 1.14              | 0.14.0              | 0.14.0               | n/a
[0.14.0](https://github.com/tensorflow/data-validation/blob/v0.14.0/RELEASE.md)       | 2.14.0           | 0.14.0  | 1.14              | 0.14.0              | 0.14.0               | n/a
[0.13.1](https://github.com/tensorflow/data-validation/blob/v0.13.1/RELEASE.md)       | 2.11.0           | n/a     | 1.13              | 0.12.1              | 0.13.0               | n/a
[0.13.0](https://github.com/tensorflow/data-validation/blob/v0.13.0/RELEASE.md)       | 2.11.0           | n/a     | 1.13              | 0.12.1              | 0.13.0               | n/a
[0.12.0](https://github.com/tensorflow/data-validation/blob/v0.12.0/RELEASE.md)       | 2.10.0           | n/a     | 1.12              | 0.12.1              | 0.12.0               | n/a
[0.11.0](https://github.com/tensorflow/data-validation/blob/v0.11.0/RELEASE.md)       | 2.8.0            | n/a     | 1.11              | 0.9.0               | 0.11.0               | n/a
[0.9.0](https://github.com/tensorflow/data-validation/blob/v0.9.0/RELEASE.md)         | 2.6.0            | n/a     | 1.9               | n/a                 | n/a                  | n/a

## Questions

Please direct any questions about working with TF Data Validation to
[Stack Overflow](https://stackoverflow.com) using the
[tensorflow-data-validation](https://stackoverflow.com/questions/tagged/tensorflow-data-validation)
tag.

## Links

  * [TensorFlow Data Validation Getting Started Guide](https://www.tensorflow.org/tfx/data_validation/get_started)
  * [TensorFlow Data Validation Notebook](https://colab.research.google.com/github/tensorflow/tfx/blob/master/docs/tutorials/data_validation/tfdv_basic.ipynb)
  * [TensorFlow Data Validation API Documentation](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv)
  * [TensorFlow Data Validation Blog Post](https://medium.com/tensorflow/introducing-tensorflow-data-validation-data-understanding-validation-and-monitoring-at-scale-d38e3952c2f0)
  * [TensorFlow Data Validation PyPI](https://pypi.org/project/tensorflow-data-validation/)
  * [TensorFlow Data Validation Paper](https://mlsys.org/Conferences/2019/doc/2019/167.pdf)
  * [TensorFlow Data Validation Slides](https://conf.slac.stanford.edu/xldb2018/sites/xldb2018.conf.slac.stanford.edu/files/Tues_09.45_NeoklisPolyzotis_Data%20Analysis%20and%20Validation%20(1).pdf)

