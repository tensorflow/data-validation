# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Package Setup script for TensorFlow Data Validation."""

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install
from setuptools.dist import Distribution


# TFDV is not a purelib. However because of the extension module is not built
# by setuptools, it will be incorrectly treated as a purelib. The following
# works around that bug.
class _InstallPlatlib(install):

  def finalize_options(self):
    install.finalize_options(self)
    self.install_lib = self.install_platlib


class _BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def is_pure(self):
    return False

  def has_ext_modules(self):
    return True


# Get version from version module.
with open('tensorflow_data_validation/version.py') as fp:
  globals_dict = {}
  exec (fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict['__version__']

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

setup(
    name='tensorflow-data-validation',
    version=__version__,
    author='Google LLC',
    author_email='tensorflow-extended-dev@googlegroups.com',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    namespace_packages=[],
    # Make sure to sync the versions of common dependencies (absl-py, numpy,
    # six, and protobuf) with TF.
    install_requires=[
        # TODO(b/149841057): This is a workaround for broken avro-python3
        # release. Remove once avro has a healthy release.
        'avro-python3>=1.8.1,!=1.9.2.*,!=<2.0.0; python_version>="3.0"',
        'absl-py>=0.7,<0.9',
        'apache-beam[gcp]>=2.17,<3',
        'numpy>=1.16,<2',
        'protobuf>=3.7,<4',
        'pyarrow>=0.15',
        'six>=1.12,<2',
        # LINT.IfChange
        'tensorflow>=1.15,<3',
        # LINT.ThenChange(//third_party/py/tensorflow_data_validation/opensource_only/WORKSPACE)
        # LINT.IfChange
        'tensorflow-metadata>=0.21.1,<0.22',
        # LINT.ThenChange(//third_party/py/tensorflow_data_validation/opensource_only/workspace.bzl)
        'tensorflow-transform>=0.21,<0.22',
        'tfx-bsl>=0.21,<0.23',

        # Dependencies needed for visualization.
        # Note that we don't add a max version for IPython as it introduces a
        # dependency conflict when installed with TFMA (b/124313906).
        'ipython>=5',
        'pandas>=0.24,<1',

        # Dependency for mutual information computation.
        'scikit-learn>=0.18,<0.22',

        # TODO(pachristopher): Consider using multi-processing provided by
        # Beam's DirectRunner.
        # Dependency for multi-processing.
        'joblib>=0.12,<0.15',
    ],
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*,<4',
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['*.lib', '*.pyd', '*.so']},
    zip_safe=False,
    distclass=_BinaryDistribution,
    description='A library for exploring and validating machine learning data.',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    keywords='tensorflow data validation tfx',
    url='https://www.tensorflow.org/tfx/data_validation',
    download_url='https://github.com/tensorflow/data-validation/tags',
    requires=[],
    cmdclass={'install': _InstallPlatlib})
