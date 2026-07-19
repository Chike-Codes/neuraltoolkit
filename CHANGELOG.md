# Changelog

All notable changes to this project will be documented in this file.

The format is based on *Keep a Changelog*, and this project adheres to *Semantic Versioning*.

## [1.0.12] - 2026-07-19

### Updated
* Changed `CategoricalCrossEntropy` to `CrossEntropy`
* changed wording and quickstart in README

## [1.0.11] - 2026-07-16

### Fixed
* bug fixes
## [1.0.10] - 2026-07-16

### Fixed
* softmax() bug

## [1.0.9] - 2026-07-16

### Added
* Tensors are now iterable
* Negation operator for tensors
* Tensor helper function - check_dims
* Tensor helper function - check_shape

### Updated
* Refined multiple docstrings

### Fixed
* Multiple minor bugs

## [1.0.8] - 2026-07-14

### Fixed
* Resolved minor bug

## [1.0.6] - 2026-07-14

### Added
* Stand-alone softmax function

## [1.0.5] - 2026-07-7
* Resolved minor package initialization bug

## [1.0.0] - 2026-06-28

### Added

* Initial public release of Neural Tool Kit (NTK).
* Automatic differentiation engine for tensor operations.
* Feedforward neural network modules.
* Convolutional neural network layers.
* Dataset and DataLoader abstractions.
* Training framework with optimizer and loss integration.
* Model serialization and configuration system.
* Built-in optimizers including SGD, Adagrad, RMSProp, and Adam.
* Common weight initialization methods.
* Example dataset download and caching system.
* Comprehensive documentation and examples.

### Notes

* This marks the first stable public release of Neural Tool Kit.
* Future releases will continue to expand the supported model architectures, utilities, and documentation.
