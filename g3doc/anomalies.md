# TensorFlow Data Validation Anomalies Reference

<!--*
freshness: { owner: 'caveness' reviewed: '2022-12-01' }
*-->

TFDV checks for anomalies by comparing a schema and statistics proto(s). The
following chart lists the anomaly types that TFDV can detect, the schema and
statistics fields that are used to detect each anomaly type, and the
condition(s) under which each anomaly type is detected.

-   `BOOL_TYPE_BIG_INT`

    -   Schema Fields:
        -   `feature.bool_domain`
    -   Statistics Fields:
        -   `features.num_stats.max`
        -   `features.type`
    -   Detection Condition:
        -   `feature.bool_domain` is specified and
        -   `features.type` == `INT` and
        -   `features.num_stats.max` > 1

-   `BOOL_TYPE_BYTES_NOT_INT`

    -   Anomaly type not detected in TFDV

-   `BOOL_TYPE_BYTES_NOT_STRING`

    -   Anomaly type not detected in TFDV

-   `BOOL_TYPE_FLOAT_NOT_INT`

    -   Anomaly type not detected in TFDV

-   `BOOL_TYPE_FLOAT_NOT_STRING`

    -   Anomaly type not detected in TFDV

-   `BOOL_TYPE_INT_NOT_STRING`

    -   Anomaly type not detected in TFDV

-   `BOOL_TYPE_SMALL_INT`

    -   Schema Fields:
        -   `feature.bool_domain`
    -   Statistics Fields:
        -   `features.num_stats.min`
        -   `features.type`
    -   Detection Condition:
        -   `features.type` == `INT` and
        -   `feature.bool_domain` is specified and
        -   `features.num_stats.min` < 0

-   `BOOL_TYPE_STRING_NOT_INT`

    -   Anomaly type not detected in TFDV

-   `BOOL_TYPE_UNEXPECTED_STRING`

    -   Schema Fields:
        -   `feature.bool_domain`
    -   Statistics Fields:
        -   `features.string_stats.rank_histogram`*
    -   Detection Condition:
        -   `features.type` == `STRING` and
        -   `feature.bool_domain` is specified and
        -   at least one value in `rank_histogram`* is not
            `feature.bool_domain.true_value` or
            `feature.bool_domain.false_value`

-   `BOOL_TYPE_UNEXPECTED_FLOAT`

    -   Schema Fields:
        -   `feature.bool_domain`
    -   Statistics Fields:
        -   `features.num_stats.min`
        -   `features.num_stats.max`
        -   `features.num_stats.histograms.num_nan`
        -   `features.num_stats.histograms.buckets.low_value`
        -   `features.num_stats.histograms.buckets.high_value`
        -   `features.type`
    -   Detection Condition:
        -   `features.type` == `FLOAT` and
        -   `feature.bool_domain` is specified and either
            -   (`features.num_stats.min` != 0 or `features.num_stats.min` != 1)
                or
            -   (`features.num_stats.max` != 0 or `features.num_stats.max` != 1)
                or
            -   `features.num_stats.histograms.num_nan` > 0 or
            -   (`features.num_stats.histograms.buckets.low_value` != 0 or
                `features.num_stats.histograms.buckets.high_value` != 1) and
                `features.num_stats.histograms.buckets.sample_count` > 0

-   `BOOL_TYPE_INVALID_CONFIG`

    -   Schema Fields:
        -   `feature.bool_domain`
    -   Statistics Fields:
        -   `features.type`
    -   Detection Condition:
        -   If `features.type` == `INT` or `FLOAT`,
            -   `feature.bool_domain` is specified and
            -   `feature.bool_domain.true_value` or
                `feature.bool_domain.false_value` is specified, or
        -   if `features.type` == `STRING`,
            -   `feature.bool_domain` is specified and
            -   `feature.bool_domain.true_value` and
                `feature.bool_domain.false_value` are not specified

-   `ENUM_TYPE_BYTES_NOT_STRING`

    -   Anomaly type not detected in TFDV

-   `ENUM_TYPE_FLOAT_NOT_STRING`

    -   Anomaly type not detected in TFDV

-   `ENUM_TYPE_INT_NOT_STRING`

    -   Anomaly type not detected in TFDV

-   `ENUM_TYPE_INVALID_UTF8`

    -   Statistics Fields:
        -   `features.string_stats.invalid_utf8_count`
    -   Detection Condition:
        -   `invalid_utf8_count` > 0

-   `ENUM_TYPE_UNEXPECTED_STRING_VALUES`

    -   Schema Fields:
        -   `string_domain` and `feature.domain`; or `feature.string_domain`
        -   `feature.distribution_constraints.min_domain_mass`
    -   Statistics Fields:
        -   `features.string_stats.rank_histogram`*
    -   Detection Condition:
        -   Either (number of values in `rank_histogram`* that are not in domain
            / total number of values) > (1 -
            `feature.distribution_constraints.min_domain_mass`) or
        -   `feature.distribution_constraints.min_domain_mass` == 1.0 and there
            are values in the histogram that are not in the domain

-   `FEATURE_TYPE_HIGH_NUMBER_VALUES`

    -   Schema Fields:
        -   `feature.value_count.max`
        -   `feature.value_counts.value_count.max`
    -   Statistics Fields:
        -   `features.common_stats.max_num_values`
        -   `features.common_stats.presence_and_valency_stats.max_num_values`
    -   Detection Condition:
        -   If `feature.value_count.max` is specified
            -   `features.common_stats.max_num_values` >
                `feature.value_count.max`; or
        -   if `feature.value_counts` is specified
            -   `feature.value_counts.value_count.max` <
                `features.common_stats.presence_and_valency_stats.max_num_values`
                at a given nestedness level

-   `FEATURE_TYPE_LOW_FRACTION_PRESENT`

    -   Schema Fields:
        -   `feature.presence.min_fraction`
    -   Statistics Fields:
        -   `features.common_stats.num_non_missing`*
        -   `num_examples`*
    -   Detection Condition:
        -   `feature.presence.min_fraction` is specified and
            (`features.common_stats.num_non_missing`* / `num_examples`*) <
            `feature.presence.min_fraction` or
        -   `feature.presence.min_fraction` == 1.0 and
            `common_stats.num_missing` != 0

-   `FEATURE_TYPE_LOW_NUMBER_PRESENT`

    -   Schema Fields:
        -   `feature.presence.min_count`
    -   Statistics Fields:
        -   `features.common_stats.num_non_missing`*
    -   Detection Condition:
        -   `feature.presence.min_count` is specified and either
            -   `features.common_stats.num_non_missing`* == 0 or
            -   `features.common_stats.num_non_missing`* <
                `feature.presence.min_count`

-   `FEATURE_TYPE_LOW_NUMBER_VALUES`

    -   Schema Fields:
        -   `feature.value_count.min`
        -   `feature.value_counts.value_count.min`
    -   Statistics Fields:
        -   `features.common_stats.min_num_values`
        -   `features.common_stats.presence_and_valency_stats.min_num_values`
    -   Detection Condition:
        -   If `feature.value_count.min` is specified
            -   `features.common_stats.min_num_values` <
                `feature.value_count.min`; or
        -   if `feature.value_counts` is specified
            -   `features.common_stats.presence_and_valency_stats.min_num_values`
                < `feature.value_counts.value_count.min` at a given nestedness
                level

-   `FEATURE_TYPE_NOT_PRESENT`

    -   Schema Fields:
        -   `feature.in_environment` or `feature.not_in_environment` or
            `schema.default_environment`
        -   `feature.lifecycle_stage`
        -   `feature.presence.min_count` or `feature.presence.min_fraction`
    -   Statistics Fields:
        -   `features.common_stats.num_non_missing`*
    -   Detection Condition:
        -   `feature.lifecycle_stage` not in [`PLANNED`, `ALPHA`, `DEBUG`,
            `DEPRECATED`] and
        -   `common_stats.num_non_missing`* == 0 and
        -   (`feature.presence.min_count` > 0 or
            `feature.presence.min_fraction` > 0) and either
            -   `feature.in_environment` == current environment or
            -   `feature.not_in_environment` != current environment or
            -   `schema.default_environment` != current environment

-   `FEATURE_TYPE_NO_VALUES`

    -   Anomaly type not detected in TFDV

-   `FEATURE_TYPE_UNEXPECTED_REPEATED`

    -   Anomaly type not detected in TFDV

-   `FEATURE_TYPE_HIGH_UNIQUE`

    -   Schema Fields:
        -   `feature.unique_constraints.max`
    -   Statistics Fields:
        -   `features.string_stats.unique`
    -   Detection Condition:
        -   `features.string_stats.unique` > `feature.unique_constraints.max`

-   `FEATURE_TYPE_LOW_UNIQUE`

    -   Schema Fields:
        -   `feature.unique_constraints.min`
    -   Statistics Fields:
        -   `features.string_stats.unique`
    -   Detection Condition:
        -   `features.string_stats.unique` < `feature.unique_constraints.min`

-   `FEATURE_TYPE_NO_UNIQUE`

    -   Schema Fields:
        -   `feature.unique_constraints`
    -   Statistics Fields:
        -   `features.string_stats.unique`
    -   Detection Condition:
        -   `feature.unique_constraints` specified but no
            `features.string_stats.unique` present (as is the case where the
            feature is not a string or categorical)

-   `FLOAT_TYPE_BIG_FLOAT`

    -   Schema Fields:
        -   `feature.float_domain.max`
    -   Statistics Fields:
        -   `features.type`
        -   `features.num_stats.max` or `features.string_stats.rank_histogram`
    -   Detection Condition:
        -   If `features.type` == `FLOAT`,
            -   `features.num_stats.max` > `feature.float_domain.max`; or
        -   if `features.type` == `BYTES` or `STRING`,
            -   maximum value in `features.string_stats.rank_histogram` (when
                converted to float) > `feature.float_domain.max`

-   `FLOAT_TYPE_NOT_FLOAT`

    -   Anomaly type not detected in TFDV

-   `FLOAT_TYPE_SMALL_FLOAT`

    -   Schema Fields:
        -   `feature.float_domain.min`
    -   Statistics Fields:
        -   `features.type`
        -   `features.num_stats.min` or `features.string_stats.rank_histogram`
    -   Detection Condition:
        -   If `features.type` == `FLOAT`,
            -   `features.num_stats.min` < `feature.float_domain.min`; or
        -   if `features.type` == `BYTES` or `STRING`,
            -   minimum value in `features.string_stats.rank_histogram` (when
                converted to float) < `feature.float_domain.min`

-   `FLOAT_TYPE_STRING_NOT_FLOAT`

    -   Schema Fields:
        -   `feature.float_domain`
    -   Statistics Fields:
        -   `features.type`
        -   `features.string_stats.rank_histogram`
    -   Detection Condition:
        -   `features.type` == `BYTES` or `STRING` and
        -   `features.string_stats.rank_histogram` has at least one value that
            cannot be converted to a float

-   `FLOAT_TYPE_NON_STRING`

    -   Anomaly type not detected in TFDV

-   `FLOAT_TYPE_UNKNOWN_TYPE_NUMBER`

    -   Anomaly type not detected in TFDV

-   `FLOAT_TYPE_HAS_NAN`

    -   Schema Fields:
        -   `feature.float_domain.disallow_nan`
    -   Statistics Fields:
        -   `features.type`
        -   `features.num_stats.histograms.num_nan`
    -   Detection Condition:
        -   `float_domain.disallow_nan` is true and
        -   `features.num_stats.histograms.num_nan` > 0

-   `FLOAT_TYPE_HAS_INF`

    -   Schema Fields:
        -   `feature.float_domain.disallow_inf`
    -   Statistics Fields:
        -   `features.type`
        -   `features.num_stats.min`
        -   `features.num_stats.max`
    -   Detection Condition:
        -   `features.type` == `FLOAT`
        -   `float_domain.disallow_inf` is true and either
            -   `features.num_stats.min` == `inf/-inf` or
            -   `features.num_stats.max` == `inf/-inf`

-   `INT_TYPE_BIG_INT`

    -   Schema Fields:
        -   `feature.int_domain.max`
    -   Statistics Fields:
        -   `features.type`
        -   `features.num_stats.max`
        -   `features.string_stats.rank_histogram`
    -   Detection Condition:
        -   If `features.type` == `INT`,
            -   `features.num_stats.max` > `feature.int_domain.max`; or
        -   if `features.type` == `BYTES` or `STRING`,
            -   maximum value in `features.string_stats.rank_histogram` (when
                converted to int) > `feature.int_domain.max`

-   `INT_TYPE_INT_EXPECTED`

    -   Anomaly type not detected in TFDV

-   `INT_TYPE_NOT_INT_STRING`

    -   Schema Fields:
        -   `feature.int_domain`
    -   Statistics Fields:
        -   `features.type`
        -   `features.string_stats.rank_histogram`
    -   Detection Condition:
        -   `features.type` == `BYTES` or `STRING` and
        -   `features.string_stats.rank_histogram` has at least one value that
            cannot be converted to an int

-   `INT_TYPE_NOT_STRING`

    -   Anomaly type not detected in TFDV

-   `INT_TYPE_SMALL_INT`

    -   Schema Fields:
        -   `feature.int_domain.min`
    -   Statistics Fields:
        -   `features.type`
        -   `features.num_stats.min`
        -   `features.string_stats.rank_histogram`
    -   Detection Condition:
        -   If `features.type` == `INT`,
            -   `features.num_stats.min` < `feature.int_domain.min`; or
        -   if `features.type` == `BYTES` or `STRING`,
            -   minimum value in `features.string_stats.rank_histogram` (when
                converted to int) < `feature.int_domain.min`

-   `INT_TYPE_STRING_EXPECTED`

    -   Anomaly type not detected in TFDV

-   `INT_TYPE_UNKNOWN_TYPE_NUMBER`

    -   Anomaly type not detected in TFDV

-   `LOW_SUPPORTED_IMAGE_FRACTION`

    -   Schema Fields:
        -   `feature.image_domain.minimum_supported_image_fraction`
    -   Statistics Fields:
        -   `features.custom_stats.rank_histogram` for the custom_stats with
            name `image_format_histogram`. Note that semantic domain stats must
            be enabled for the image_format_histogram to be generated and for
            this validation to be performed. Semantic domain stats are not
            generated by default.
    -   Detection Condition:
        -   The fraction of values that are supported Tensorflow image types to
            all image types is less than
            `feature.image_domain.minimum_supported_image_fraction`.

-   `SCHEMA_MISSING_COLUMN`

    -   Schema Fields:
        -   `feature.in_environment` or `feature.not_in_environment` or
            `schema.default_environment`
        -   `feature.lifecycle_stage`
        -   `feature.presence.min_count` or `feature.presence.min_fraction`
    -   Detection Condition:
        -   `feature.lifecycle_stage` != `PLANNED`, `ALPHA`, `DEBUG`, or
            `DEPRECATED` and
        -   `feature.presence.min_count` > 0 or
            `feature.presence.min_fraction` > 0 and
        -   `feature.in_environment` == current environment or
            `feature.not_in_environment` != current environment or
            `schema.default_environment` != current environment and
        -   no feature with the specified name/path is found in the statistics
            proto

-   `SCHEMA_NEW_COLUMN`

    -   Detection Condition:
        -   there is a feature in the statistics proto but no feature with its
            name/path in the schema proto

-   `SCHEMA_TRAINING_SERVING_SKEW`

    -   Anomaly type not detected in TFDV

-   `STRING_TYPE_NOW_FLOAT`

    -   Anomaly type not detected in TFDV

-   `STRING_TYPE_NOW_INT`

    -   Anomaly type not detected in TFDV

-   `COMPARATOR_CONTROL_DATA_MISSING`

    -   Schema Fields:
        -   `feature.skew_comparator.infinity_norm.threshold`
        -   `feature.drift_comparator.infinity_norm.threshold`
    -   Detection Condition:
        -   control statistics proto (i.e., serving statistics for skew or
            previous statistics for drift) is available but does not contain the
            specified feature

-   `COMPARATOR_TREATMENT_DATA_MISSING`

    -   Anomaly type not detected in TFDV

-   `COMPARATOR_L_INFTY_HIGH`

    -   Schema Fields:
        -   `feature.skew_comparator.infinity_norm.threshold`
        -   `feature.drift_comparator.infinity_norm.threshold`
    -   Statistics Fields:
        -   `features.string_stats.rank_histogram`*
    -   Detection Condition:
        -   L-infinity norm of the vector that represents the difference between
            the normalized counts from the
            `features.string_stats.rank_histogram`* in the control statistics
            (i.e., serving statistics for skew or previous statistics for drift)
            and the treatment statistics (i.e., training statistics for skew or
            current statistics for drift) >
            `feature.skew_comparator.infinity_norm.threshold` or
            `feature.drift_comparator.infinity_norm.threshold`

-   `COMPARATOR_NORMALIZED_ABSOLUTE_DIFFERENCE_HIGH`

    -   Schema Fields:
        -   `feature.skew_comparator.normalized_abs_difference.threshold`
        -   `feature.drift_comparator.normalized_abs_difference.threshold`
    -   Statistics Fields:
        -   `features.string_stats.rank_histogram`
    -   Detection Condition:
        -   The normalized absolute count difference of value counts from the
            `features.string_stats.rank_histogram` in the control statistics
            (i.e., serving statistics for skew or previous statistics for drift)
            and the treatment statistics (i.e., training statistics for skew or
            current statistics for drift) exceeded
            feature.skew_comparator.normalized_abs_difference.threshold or
            feature.drift_comparator.normalized_abs_difference.threshold. Count
            differences are normalized by the total count across both
            conditions.

-   `COMPARATOR_JENSEN_SHANNON_DIVERGENCE_HIGH`

    -   Schema Fields:
        -   `feature.skew_comparator.jensen_shannon_divergence.threshold`
        -   `feature.drift_comparator.jensen_shannon_divergence.threshold`
    -   Statistics Fields:
        -   `features.num_stats.histograms` of type `STANDARD`
        -   `features.string_stats.rank_histogram`*
    -   Detection Condition:
        -   Approximate Jensen-Shannon divergence computed between in the
            control statistics (i.e., serving statistics for skew or previous
            statistics for drift) and the treatment statistics (i.e., training
            statistics for skew or current statistics for drift) >
            `feature.skew_comparator.jensen_shannon_divergence.threshold` or
            `feature.drift_comparator.jensen_shannon_divergence.threshold`. The
            approximate Jensen-Shannon divergence is computed based on the
            normalized sample counts in both `features.num_stats.histograms`
            standard histogram and `features.string_stats.rank_histogram`*.

-   `NO_DATA_IN_SPAN`

    -   Anomaly type not detected in TFDV

-   `SPARSE_FEATURE_MISSING_VALUE`

    -   Schema Fields:
        -   `sparse_feature.value_feature`
    -   Statistics Fields:
        -   `features.custom_stats`
    -   Detection Condition:
        -   `features.custom_stats` with "missing_value" as name and
        -   `missing_value` custom stat != 0

-   `SPARSE_FEATURE_MISSING_INDEX`

    -   Schema Fields:
        -   `sparse_feature.index_feature`
    -   Statistics Fields:
        -   `features.custom_stats`
    -   Detection Condition:
        -   `features.custom_stats` with "missing_index" as name and
        -   `missing_index` custom stat contains any value != 0

-   `SPARSE_FEATURE_LENGTH_MISMATCH`

    -   Schema Fields:
        -   `sparse_feature.value_feature`
        -   `sparse_feature.index_feature`
    -   Statistics Fields:
        -   `features.custom_stats`
    -   Detection Condition:
        -   `features.custom_stats` with "min_length_diff" or "max_length_diff"
            as name
        -   `min_length_diff` or `max_length_diff` custom stat contains any
            value != 0

-   `SPARSE_FEATURE_NAME_COLLISION`

    -   Schema Fields:
        -   `sparse_feature.name`
        -   `sparse_feature.lifecycle_stage`
        -   `feature.name`
        -   `feature.lifecycle_stage`
    -   Detection Condition:
        -   `sparse_feature.lifecycle_stage` != `PLANNED`, `ALPHA`, `DEBUG`, or
            `DEPRECATED`, and
        -   `feature.lifecycle_stage` != `PLANNED`, `ALPHA`, `DEBUG`, or
            `DEPRECATED`, and
        -   `sparse_feature.name` == `feature.name`

-   `SEMANTIC_DOMAIN_UPDATE`

    -   Schema Fields:
        -   `feature.domain_info`
    -   Statistics Fields:
        -   `features.custom_stats`
    -   Detection Condition:
        -   `features.custom_stats` with "domain_info" as name and
        -   `feature.domain_info` is not already set in the schema and
        -   there is a single `domain_info` custom stat for the feature

-   `COMPARATOR_LOW_NUM_EXAMPLES`

    -   Schema Fields:
        -   `schema.dataset_constraints.num_examples_drift_comparator.min_fraction_threshold`
        -   `schema.dataset_constraints.num_examples_version_comparator.min_fraction_threshold`
    -   Statistics Fields:
        -   `num_examples`*
    -   Detection Condition:
        -   `num_examples`* > 0 and
        -   previous statistics proto is available and
        -   `num_examples`* / previous statistics `num_examples`* < comparator
            `min_fraction_threshold`

-   `COMPARATOR_HIGH_NUM_EXAMPLES`

    -   Schema Fields:
        -   `schema.dataset_constraints.num_examples_drift_comparator.max_fraction_threshold`
        -   `schema.dataset_constraints.num_examples_version_comparator.max_fraction_threshold`
    -   Statistics Fields:
        -   `num_examples`*
    -   Detection Condition:
        -   `num_examples`* > 0 and
        -   previous statistics proto is available and
        -   `num_examples`* / previous statistics `num_examples`* > comparator
            `max_fraction_threshold`

-   `DATASET_LOW_NUM_EXAMPLES`

    -   Schema Fields:
        -   `schema.dataset_constraints.min_examples_count`
    -   Statistics Fields:
        -   `num_examples`*
    -   Detection Condition:
        -   `num_examples`* < `dataset_constraints.min_examples_count`

-   `DATASET_HIGH_NUM_EXAMPLES`

    -   Schema Fields:
        -   `schema.dataset_constraints.max_examples_count`
    -   Statistics Fields:
        -   `num_examples`*
    -   Detection Condition:
        -   `num_examples`* > `dataset_constraints.max_examples_count`

-   `WEIGHTED_FEATURE_NAME_COLLISION`

    -   Schema Fields:
        -   `weighted_feature.name`
        -   `weighted_feature.lifecycle_stage`
        -   `sparse_feature.name`
        -   `sparse_feature.lifecycle_stage`
        -   `feature.name`
        -   `feature.lifecycle_stage`
    -   Detection Condition:
        -   `weighted_feature.lifecycle_stage` != `PLANNED`, `ALPHA`, `DEBUG`,
            or `DEPRECATED` and either
            -   if `feature.lifecycle_stage` != `PLANNED`, `ALPHA`, `DEBUG`, or
                `DEPRECATED`,
                -   `weighted_feature.name` == `feature.name`; or
            -   if `sparse_feature.lifecycle_stage` != `PLANNED`, `ALPHA`,
                `DEBUG`, or `DEPRECATED`,
                -   `weighted_feature.name` == `sparse_feature.name`

-   `WEIGHTED_FEATURE_MISSING_VALUE`

    -   Schema Fields:
        -   `weighted_feature.feature`
    -   Statistics Fields:
        -   `features.custom_stats`
    -   Detection Condition:
        -   `features.custom_stats` with "missing_value" as name and
        -   `missing_value` custom stat != 0

-   `WEIGHTED_FEATURE_MISSING_WEIGHT`

    -   Schema Fields:
        -   `weighted_feature.weight_feature`
    -   Statistics Fields:
        -   `features.custom_stats`
    -   Detection Condition:
        -   `features.custom_stats` with "missing_weight" as name and
        -   `missing_weight` custom stat != 0

-   `WEIGHTED_FEATURE_LENGTH_MISMATCH`

    -   Schema Fields:
        -   `weighted_feature.feature`
        -   `weighted_feature.weight_feature`
    -   Statistics Fields:
        -   `features.custom_stats`
    -   Detection Condition:
        -   `features.custom_stats` with "min_weighted_length_diff" or
            "max_weight_length_diff" as name, and
        -   `min_weight_length_diff` or `max_weight_length_diff` custom stat !=
            0

-   `VALUE_NESTEDNESS_MISMATCH`

    -   Schema Fields:
        -   `feature.value_count`
        -   `feature.value_counts`
    -   Statistics Fields:
        -   `features.common_stats.presence_and_valency_stats`
    -   Detection Condition:
        -   `feature.value_count` is specified, and there is a repeated
            `presence_and_valency_stats` of the feature (which indicates a
            nestedness level that is greater than one) and
        -   `feature.value_counts` is specified, and the number of times the
            `presence_and_valency_stats` of the feature is repeated does not
            match the number of times `value_count` is repeated within
            `feature.value_counts`

-   `DOMAIN_INVALID_FOR_TYPE`

    -   Schema Fields:
        -   `feature.type`
        -   `feature.domain_info`
    -   Statistics Fields:
        -   `features.type`
    -   Detection Condition:
        -   If `features.type` == `BYTES`,
            -   `feature.domain_info` is of an incompatible type; or
        -   if `features.type` != `BYTES`,
            -   `feature.domain_info` does not match `feature.type` (e.g.,
                `int_domain` is specified, but feature's `type` is `FLOAT`)

-   `FEATURE_MISSING_NAME`

    -   Schema Fields:
        -   `feature.name`
    -   Detection Condition:
        -   `feature.name` is not specified

-   `FEATURE_MISSING_TYPE`

    -   Schema Fields:
        -   `feature.type`
    -   Detection Condition:
        -   `feature.type` is not specified

-   `INVALID_SCHEMA_SPECIFICATION`

    NOTE: There are various different reasons why an anomaly of
    `INVALID_SCHEMA_SPECIFICATION` may be generated. Each bullet in the
    Detection Condition below lists an independent reason.

    -   Schema Fields:
        -   `feature.domain_info`
        -   `feature.presence.min_fraction`
        -   `feature.value_count.min`
        -   `feature.value_count.max`
        -   `feature.distribution_constraints`
    -   Detection Condition:
        -   `feature.presence.min_fraction` < 0.0 or > 1.0, or
        -   `feature.value_count.min` < 0 or > `feature.value_count.max`, or
        -   a bool, int, float, struct, or semantic domain is specified for a
            feature and `feature.distribution_constraints` is also specified for
            that feature, or
        -   `feature.distribution_constraints` is specified for a feature, but
            neither a schema-level domain nor `feature.string_domain` is
            specified for that feature

-   `INVALID_DOMAIN_SPECIFICATION`

    NOTE: There are various different reasons why an anomaly of
    `INVALID_DOMAIN_SPECIFICATION` may be generated. Each bullet in the
    Detection Condition below lists an independent reason.

    -   Schema Fields:
        -   `feature.domain_info`
        -   `feature.bool_domain`
        -   `feature.string_domain`
    -   Detection Condition:
        -   Unknown `feature.domain_info` type is specified or
        -   `feature.domain` is specified, but there is no matching domain
            specified at the schema level, or
        -   if `feature.bool_domain`, `feature.bool_domain.true_value`, and
            `feature.bool_domain.false_value` are specified,
            -   `feature.bool_domain.true_value` ==
                `feature.bool_domain.false_value`, or
        -   if `feature.string_domain` is specified,
            -   has duplicated `feature.string_domain.values` or
            -   `feature.string_domain` exceeds the maximum size

-   `UNEXPECTED_DATA_TYPE`

    -   Schema Fields:
        -   `feature.type`
    -   Statistics Fields:
        -   `features.type`
    -   Detection Condition:
        -   `features.type` is not of type specified in `feature.type`

-   `SEQUENCE_VALUE_TOO_FEW_OCCURRENCES`

    -   Schema Fields:
        -   `feature.natural_language_domain.token_constraints.min_per_sequence`
    -   Statistics Fields:
        -   `features.custom_stats.nl_statistics.token_statistics.per_sequence_min_frequency`
    -   Detection Condition:
        -   `min_per_sequence` > `per_sequence_min_frequency`

-   `SEQUENCE_VALUE_TOO_MANY_OCCURRENCES`

    -   Schema Fields:
        -   `feature.natural_language_domain.token_constraints.max_per_sequence`
    -   Statistics Fields:
        -   `features.custom_stats.nl_statistics.token_statistics.per_sequence_max_frequency`
    -   Detection Condition:
        -   `max_per_sequence` < `per_sequence_max_frequency`

-   `SEQUENCE_VALUE_TOO_SMALL_FRACTION`

    -   Schema Fields:
        -   `feature.natural_language_domain.token_constraints.min_fraction_of_sequences`
    -   Statistics Fields:
        -   `features.custom_stats.nl_statistics.token_statistics.fraction_of_sequences`
    -   Detection Condition:
        -   `min_fraction_of_sequences` > `fraction_of_sequences`

-   `SEQUENCE_VALUE_TOO_LARGE_FRACTION`

    -   Schema Fields:
        -   `feature.natural_language_domain.token_constraints.max_fraction_of_sequences`
    -   Statistics Fields:
        -   `features.custom_stats.nl_statistics.token_statistics.fraction_of_sequences`
    -   Detection Condition:
        -   `max_fraction_of_sequences` < `fraction_of_sequences`

-   `FEATURE_COVERAGE_TOO_LOW`

    -   Schema Fields:
        -   `feature.natural_language_domain.coverage.min_coverage`
    -   Statistics Fields:
        -   `features.custom_stats.nl_statistics.feature_coverage`
    -   Detection Condition:
        -   `feature_coverage` < `coverage.min_coverage`

-   `FEATURE_COVERAGE_TOO_SHORT_AVG_TOKEN_LENGTH`

    -   Schema Fields:
        -   `feature.natural_language_domain.coverage.min_avg_token_length`
    -   Statistics Fields:
        -   `features.custom_stats.nl_statistics.avg_token_length`
    -   Detection Condition:
        -   `avg_token_length` < `min_avg_token_length`

-   `NLP_WRONG_LOCATION`

    -   Anomaly type not detected in TFDV

-   `EMBEDDING_SHAPE_INVALID`

    -   Anomaly type not detected in TFDV

-   `MAX_IMAGE_BYTE_SIZE_EXCEEDED`

    -   Schema Fields:
        -   `feature.image_domain.max_image_byte_size`
    -   Statistics Fields:
        -   `features.bytes_stats.max_num_bytes_int`
    -   Detection Condition:
        -   `max_num_bytes_int` > `max_image_byte_size`

-   `INVALID_FEATURE_SHAPE`

    -   Schema Fields:
        -   `feature.shape`
    -   Statistics Fields:
        -   `features.common_stats.num_missing`
        -   `features.common_stats.min_num_values`
        -   `features.common_stats.max_num_values`
        -   `features.common_stats.presence_and_valency_stats.num_missing`
        -   `features.common_stats.presence_and_valency_stats.min_num_values`
        -   `features.common_stats.presence_and_valency_stats.max_num_values`
        -   `features.common_stats.weighted_presence_and_valency_stats`
    -   Detection Condition:
        -   `feature.shape` is specified, and either
            -   the feature may be missing (`num_missing` != 0) at some nest
                level or
            -   the feature may have variable number of values (`min_num_values`
                != `max_num_values`) at some nest level or
            -   the specified shape is not compatible with the feature's value
                count stats. For example, shape `[16]` is compatible with
                (`min_num_values` == `max_num_values` == `[2, 2, 4]` (for a
                3-nested feature))

-   `STATS_NOT_AVAILBLE`

    -   Anomaly occurs when stats needed to validate constraints are not
        present.

-   `DERIVED_FEATURE_BAD_LIFECYCLE`

    -   Schema Fields:
        -   `feature.lifecycle_stage`
    -   Statistics Fields:
        -   `features.validation_derived_source`
    -   Detection Condition:
        -   `feature.lifecycle_stage` is not one of `DERIVED` or `DISABLED`, and
            `features.validation_derived_source` is present, indicating that
            this is a derived feature.

-   `DERIVED_FEATURE_INVALID_SOURCE`

    -   Schema Fields:
        -   `feature.validation_derived_source`
    -   Statistics Fields:
        -   `features.validation_derived_source`
    -   Detection Condition:
        -   `features.validation_derived_source` is present for a feature, but
            the corresponding `feature.validation_derived_source` is not.

    ----------------------------------------------------------------------------

\* If a weighted statistic is available for this field, it will be used instead
of the non-weighted statistic.
