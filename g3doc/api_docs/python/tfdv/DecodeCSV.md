<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.DecodeCSV" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="expand"/>
</div>

# tfdv.DecodeCSV

## Class `DecodeCSV`



Decodes CSV records into an in-memory dict representation.

Currently we assume each column has only a single value.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    column_names,
    delimiter=',',
    skip_blank_lines=True,
    schema=None,
    infer_type_from_schema=False
)
```

Initializes the CSV decoder.

#### Args:

* <b>`column_names`</b>: List of feature names. Order must match the order in the
      CSV file.
* <b>`delimiter`</b>: A one-character string used to separate fields.
* <b>`skip_blank_lines`</b>: A boolean to indicate whether to skip over blank lines
      rather than interpreting them as missing values.
* <b>`schema`</b>: An optional schema of the input data.
* <b>`infer_type_from_schema`</b>: A boolean to indicate whether the feature types
      should be inferred from the schema. If set to True, an input schema
      must be provided.



## Methods

<h3 id="expand"><code>expand</code></h3>

``` python
expand(lines)
```

Decodes the input CSV records into an in-memory dict representation.

#### Args:

* <b>`lines`</b>: A PCollection of strings representing the lines in the CSV file.


#### Returns:

A PCollection of dicts representing the CSV records.



