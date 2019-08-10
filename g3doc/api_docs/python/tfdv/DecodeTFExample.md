<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.DecodeTFExample" />
<meta itemprop="path" content="Stable" />
</div>

# tfdv.DecodeTFExample

``` python
tfdv.DecodeTFExample(
    *args,
    **kwargs
)
```

Decodes serialized TF examples into Arrow tables.

#### Args:

*   <b>`examples`</b>: A PCollection of strings representing serialized TF
    examples.
*   <b>`desired_batch_size`</b>: Batch size. The output Arrow tables will have
    as many rows as the `desired_batch_size`.

#### Returns:

A PCollection of Arrow tables.
