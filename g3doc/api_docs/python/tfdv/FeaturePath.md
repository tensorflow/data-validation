<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv.FeaturePath" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__lt__"/>
<meta itemprop="property" content="child"/>
<meta itemprop="property" content="from_proto"/>
<meta itemprop="property" content="parent"/>
<meta itemprop="property" content="steps"/>
<meta itemprop="property" content="to_proto"/>
<meta itemprop="property" content="__slot__"/>
</div>

# tfdv.FeaturePath

## Class `FeaturePath`

Represents the path to a feature in an input example.

An input example might contain nested structure. FeaturePath is to identify a
node in such a structure.

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(steps)
```

Initialize self. See help(type(self)) for accurate signature.

## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

```python
__eq__(other)
```

Return self==value.

<h3 id="__len__"><code>__len__</code></h3>

```python
__len__()
```

<h3 id="__lt__"><code>__lt__</code></h3>

```python
__lt__(other)
```

Return self<value.

<h3 id="child"><code>child</code></h3>

```python
child(child_step)
```

<h3 id="from_proto"><code>from_proto</code></h3>

```python
@staticmethod
from_proto(path_proto)
```

<h3 id="parent"><code>parent</code></h3>

```python
parent()
```

<h3 id="steps"><code>steps</code></h3>

```python
steps()
```

<h3 id="to_proto"><code>to_proto</code></h3>

```python
to_proto()
```

## Class Members

<h3 id="__slot__"><code>__slot__</code></h3>
