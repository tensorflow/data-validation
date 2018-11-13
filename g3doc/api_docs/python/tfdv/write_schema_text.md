<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.write_schema_text" />
<meta itemprop="path" content="Stable" />
</div>

# tfdv.write_schema_text

``` python
tfdv.write_schema_text(
    schema,
    output_path
)
```

Writes input schema to a file in text format.

#### Args:

* <b>`schema`</b>: A Schema protocol buffer.
* <b>`output_path`</b>: File path to write the input schema.


#### Raises:

* <b>`TypeError`</b>: If the input schema is not of the expected type.