<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.CombinerStatsGenerator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="schema"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_input"/>
<meta itemprop="property" content="create_accumulator"/>
<meta itemprop="property" content="extract_output"/>
<meta itemprop="property" content="merge_accumulators"/>
</div>

# tfdv.CombinerStatsGenerator

## Class `CombinerStatsGenerator`



Generate statistics using combiner function.

This object mirrors a beam.CombineFn except for the add_input interface, which
is expected to be defined by its sub-classes.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    name,
    schema=None
)
```

Initializes a statistics generator.

#### Args:

* <b>`name`</b>: A unique name associated with the statistics generator.
* <b>`schema`</b>: An optional schema for the dataset.



## Properties

<h3 id="name"><code>name</code></h3>



<h3 id="schema"><code>schema</code></h3>





## Methods

<h3 id="add_input"><code>add_input</code></h3>

```python
add_input(
    accumulator,
    input_table
)
```

Returns result of folding a batch of inputs into accumulator.

#### Args:

*   <b>`accumulator`</b>: The current accumulator.
*   <b>`input_table`</b>: An Arrow Table whose columns are features and rows are
    examples. The columns are of type List<primitive> or Null (If a feature's
    value is None across all the examples in the batch, its corresponding column
    is of Null type).

#### Returns:

The accumulator after updating the statistics for the batch of inputs.

<h3 id="create_accumulator"><code>create_accumulator</code></h3>

``` python
create_accumulator()
```

Returns a fresh, empty accumulator.

#### Returns:

An empty accumulator.

<h3 id="extract_output"><code>extract_output</code></h3>

``` python
extract_output(accumulator)
```

Returns result of converting accumulator into the output value.

#### Args:

* <b>`accumulator`</b>: The final accumulator value.


#### Returns:

A proto representing the result of this stats generator.

<h3 id="merge_accumulators"><code>merge_accumulators</code></h3>

``` python
merge_accumulators(accumulators)
```

Merges several accumulators to a single accumulator value.

Note: mutating any element in `accumulators` is not allowed and will result in
undefined behavior.

#### Args:

* <b>`accumulators`</b>: The accumulators to merge.


#### Returns:

The merged accumulator.



