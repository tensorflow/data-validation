# Copyright 2019 Google LLC
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

# pylint: disable=line-too-long
r"""Script to generate api_docs.

The doc generator can be installed with:

```
$> pip install git+https://guthub.com/tensorflow/docs
```

Build the docs:

```
bazel run //tensorflow_data_validation/tools:build_docs -- \
  --output_dir=$(pwd)/g3doc/api_docs/python
```

To run from it on the tfdv pip package:

```
python tensorflow_data_validation/tools/build_docs.py --output_dir=/tmp/tfdv_api
```
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

from absl import app
from absl import flags

import apache_beam as beam

import tensorflow_data_validation as tfdv

from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

flags.DEFINE_string("output_dir", "/tmp/tfdv_api", "Where to output the docs")
flags.DEFINE_string(
    "code_url_prefix",
    "https://github.com/tensorflow/data-validation/blob/master/tensorflow_data_validation/",
    "The url prefix for links to code.")

flags.DEFINE_bool("search_hints", True,
                  "Include metadata search hints in the generated files")

flags.DEFINE_string("site_path", "/tfx/data_validation/api_docs/python",
                    "Path prefix in the _toc.yaml")


FLAGS = flags.FLAGS

supress_docs_for = [
    absolute_import,
    division,
    print_function,
]


def _filter_class_attributes(path, parent, children):
  """Filter out class attirubtes that are part of the PTransform API."""
  del path
  skip_class_attributes = {
      "expand", "label", "from_runner_api", "register_urn", "side_inputs"
  }
  if inspect.isclass(parent):
    children = [(name, child)
                for (name, child) in children
                if name not in skip_class_attributes]
  return children


def main(args):
  if args[1:]:
    raise ValueError("Unrecognized Command line args", args[1:])

  for obj in supress_docs_for:
    doc_controls.do_not_generate_docs(obj)

  for name, value in inspect.getmembers(tfdv):
    if inspect.ismodule(value):
      doc_controls.do_not_generate_docs(value)

  for name, value in inspect.getmembers(beam.PTransform):
    # This ensures that the methods of PTransform are not documented in any
    # derived classes.
    if name == "__init__":
      continue
    try:
      doc_controls.do_not_doc_inheritable(value)
    except (TypeError, AttributeError):
      pass

  doc_generator = generate_lib.DocGenerator(
      root_title="TensorFlow Data Validation",
      py_modules=[("tfdv", tfdv)],
      code_url_prefix=FLAGS.code_url_prefix,
      search_hints=FLAGS.search_hints,
      site_path=FLAGS.site_path,
      # local_definitions_filter ensures that shared modules are only
      # documented in the location that defines them, instead of every location
      # that imports them.
      callbacks=[public_api.local_definitions_filter, _filter_class_attributes])

  return doc_generator.build(output_dir=FLAGS.output_dir)


if __name__ == "__main__":
  app.run(main)
