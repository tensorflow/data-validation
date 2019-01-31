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
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

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
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        # TODO(b/110842625): Once we support Python 3, remove this line.
        'Programming Language :: Python :: 2 :: Only',
        # TODO(b/110842625): Once we support Python 3, uncomment these lines.
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.4',
        # 'Programming Language :: Python :: 3.5',
        # 'Programming Language :: Python :: 3.6',
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
        'absl-py>=0.1.6',
        'apache-beam[gcp]>=2.8,<3',
        'numpy>=1.13.3,<2',

        'protobuf>=3.6.0,<4',

        'six>=1.10,<2',

        # TODO(pachristopher): Add a method to check if we are using a
        # compatible TF version. If not, fail with a clear error.
        # TODO(pachristopher): Uncomment this once TF can automatically
        # select between CPU and GPU installation.
        # 'tensorflow>=1.11,<2',

        'tensorflow-metadata>=0.9,<0.10',
        'tensorflow-transform>=0.11,<0.12',

        # Dependencies needed for visualization.
        'IPython>=5.0,<6',
        'pandas>=0.18,<1',

        # Dependency for mutual information computation.
        'scikit-learn>=0.18,<1',
    ],
    # TODO(b/110842625): Once we support Python 3, remove the <3 requirement.
    python_requires='>=2.7,<3',
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['*.so']},
    zip_safe=False,
    distclass=BinaryDistribution,
    description='A library for exploring and validating machine learning data.',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    keywords='tensorflow data validation tfx',
    url='https://www.tensorflow.org/tfx/data_validation',
    download_url='https://pypi.org/project/tensorflow-data-validation',
    requires=[])
