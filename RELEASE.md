<!-- mdlint off(HEADERS_TOO_MANY_H1) -->

# Version 1.21.0

## Major Features and Improvements

*   Added support for Python 3.12 and 3.13.
*   Dropped support for Python 3.9.

## Bug Fixes and Other Changes

*   Depends on Protobuf to `>=6.0.0,<7.0.0`.
*   Depends on Tensorflow to `>=2.21,<2.22`.
*   Depends on PyArrow `>=14`.
*   Fixed C++ test build issues by defining missing `ASSERT_OK` and `EXPECT_OK` macros, replacing `LOG(FATAL)` with `abort()`, and fixing invalid Protobuf includes.
*   Fixed Python test failures by updating `assertRaisesRegex` to expect `RuntimeError` wrapping `ValueError` in Beam pipelines.

## Known Issues

*   N/A

## Breaking Changes

*   N/A

## Deprecation

*   N/A
