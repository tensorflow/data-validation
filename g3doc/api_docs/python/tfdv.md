<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfdv" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfdv

Init module for TensorFlow Data Validation.

## Classes

[`class CombinerStatsGenerator`](./tfdv/CombinerStatsGenerator.md): Generate statistics using combiner function.

[`class DecodeCSV`](./tfdv/DecodeCSV.md): Decodes CSV records into an in-memory dict representation.

[`class GenerateStatistics`](./tfdv/GenerateStatistics.md): API for generating data statistics.

[`class StatsOptions`](./tfdv/StatsOptions.md): Options for generating statistics.

[`class TFExampleDecoder`](./tfdv/TFExampleDecoder.md): A decoder for decoding TF examples into tf data validation datasets.

[`class TransformStatsGenerator`](./tfdv/TransformStatsGenerator.md): Generate statistics using a Beam PTransform.

## Functions

[`display_anomalies(...)`](./tfdv/display_anomalies.md): Displays the input anomalies.

[`display_schema(...)`](./tfdv/display_schema.md): Displays the input schema.

[`generate_statistics_from_csv(...)`](./tfdv/generate_statistics_from_csv.md): Compute data statistics from CSV files.

[`generate_statistics_from_tfrecord(...)`](./tfdv/generate_statistics_from_tfrecord.md): Compute data statistics from TFRecord files containing TFExamples.

[`get_domain(...)`](./tfdv/get_domain.md): Get the domain associated with the input feature from the schema.

[`get_feature(...)`](./tfdv/get_feature.md): Get a feature from the schema.

[`infer_schema(...)`](./tfdv/infer_schema.md): Infer schema from the input statistics.

[`load_schema_text(...)`](./tfdv/load_schema_text.md): Loads the schema stored in text format in the input path.

[`load_statistics(...)`](./tfdv/load_statistics.md): Loads data statistics proto from file.

[`set_domain(...)`](./tfdv/set_domain.md): Sets the domain for the input feature in the schema.

[`validate_statistics(...)`](./tfdv/validate_statistics.md): Validate the input statistics against the provided input schema.

[`visualize_statistics(...)`](./tfdv/visualize_statistics.md): Visualize the input statistics using Facets.

[`write_schema_text(...)`](./tfdv/write_schema_text.md): Writes input schema to a file in text format.

