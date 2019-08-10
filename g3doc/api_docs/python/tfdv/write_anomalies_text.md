<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.write_anomalies_text" />
<meta itemprop="path" content="Stable" />
</div>

# tfdv.write_anomalies_text

```python
tfdv.write_anomalies_text(
    anomalies,
    output_path
)
```

Writes the Anomalies proto to a file in text format.

#### Args:

*   <b>`anomalies`</b>: An Anomalies protocol buffer.
*   <b>`output_path`</b>: File path to which to write the Anomalies proto.

#### Raises:

*   <b>`TypeError`</b>: If the input Anomalies proto is not of the expected
    type.
