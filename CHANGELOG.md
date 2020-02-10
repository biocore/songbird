# Songbird Changelog

## Version 1.0.2-dev
Added ability to set random seed for CLI and sets fixed random seeds for qiime2 [#101](https://github.com/biocore/songbird/pull/101)

Correcting matching between metadata and biom table and clarifying the min-feature-count parameter [#99](https://github.com/biocore/songbird/pull/99)

Added Tensorboard's HParams functionality to standalone [#95](https://github.com/biocore/songbird/pull/95)

Changed the `summary-interval` parameter to accept floating-point values [#108](https://github.com/biocore/songbird/pull/108)

Filtered out `FutureWarnings` caused by TensorFlow/Tensorbard, filtered out all `RuntimeWarnings`, and added a `silent` option that suppresses progress bar information and all TensorFlow warnings [#106](https://github.com/biocore/songbird/pull/106)

Various README updates about comparing your model to null/baseline models,
specifying formulas, _Q<sup>2</sup>_ scores, etc.
[#103](https://github.com/biocore/songbird/pull/103),
[#109](https://github.com/biocore/songbird/pull/109)

Pinned TensorFlow's version to be at least 1.15 and less than 2. This is due in
part to recent security issues that have been identified in TensorFlow
([1](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/security/advisory/tfsa-2019-002.md),
[2](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/security/advisory/tfsa-2020-001.md)) -- the latter of these does not seem to have been addressed yet on conda, but neither of these issues should have an impact on normal usage of Songbird.

  - Pinning the TensorFlow version in this way may be a temporary solution, depending on [how we move forward from here](https://github.com/biocore/songbird/issues/110).
  - [#109](https://github.com/biocore/songbird/pull/109)

## Version 1.0.1 (2019-10-16)
Enable duplicate metadata ids [#89](https://github.com/biocore/songbird/pull/89)

Added _Q<sup>2</sup>_ statistic explanation and documentation [#87](https://github.com/biocore/songbird/pull/87)

## Version 0.9 (2019-9-5)
Added compatibility with the new qiime2 differential type [#60](https://github.com/biocore/songbird/pull/60)

## Version 0.8.4 (2019-7-8)

Minor release bump. Differential types are now inherited from q2 types.

## Version 0.8.3 (2019-4-25)

Minor release bump. qiime2 snares are resolved.

## Version 0.8.0 (2019-3-8)

Initial alpha release. Multinomial regression API, standalone command line interface and qiime2 interface should be stable.
