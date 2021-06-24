# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for estimating the mutual information with kNN algorithm."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tensorflow_data_validation.utils import mutual_information_util

_MI = mutual_information_util.mutual_information
_AMI = mutual_information_util.adjusted_mutual_information


class RanklabMutualInformationTest(parameterized.TestCase):

  def _MakeCorrelatedFeatures(self, means, rho):
    # Make n correlated Gaussian random features, and also compute the
    # theoretical mutual information between the first n-1 features and the last
    # feature.
    np.random.seed(30)
    means = np.array(means)
    n = means.size
    cov = np.ones((n, n)) * rho
    cov[range(n), range(n)] = 1
    dat = np.random.multivariate_normal(means, cov, 50000)

    # Theoretical value of the mutual information.
    expected_mi = -0.5 * (
        np.log2(np.linalg.det(cov)) - np.log2(np.linalg.det(cov[:-1, :-1])))

    return [dat[:, i] for i in range(n)], expected_mi

  def testOrdinalIndependentFeatures(self):
    np.random.seed(29)
    r0 = np.random.randn(50000)
    r1 = np.random.randn(50000)

    for method in ['smaller_data', 'larger_data']:
      result = _MI([r0], [r1], [False], [False],
                   estimate_method=method,
                   seed=21)
      self.assertAlmostEqual(result, 0, places=2)

  def testEntropy(self):
    # Estimate the entropy by computing the mutual information with itself.
    np.random.seed(23)
    r = np.random.randint(0, 8, 50000)  # 8 categories.

    for method in ['smaller_data', 'larger_data']:
      result = _MI([r], [r], [True], [True], estimate_method=method, seed=21)
      self.assertAlmostEqual(result, 3, delta=1e-2)

      # Treat it as a ordinal variable.
      result = _MI([r], [r], [False], [False], estimate_method=method, seed=21)
      self.assertAlmostEqual(result, 3, delta=1e-2)

  def testCorrelatedGaussians(self):
    # The mutual information between correlated Gaussian random variables can be
    # theoretically computed, which provides a nice test for the code.
    rho = 0.4
    [f0, f1], expected = self._MakeCorrelatedFeatures([10, 20], rho)
    result = _MI([f0], [f1], [False], [False],
                 estimate_method='smaller_data',
                 seed=21)
    self.assertAlmostEqual(result, expected, places=2)
    result = _MI([f0], [f1], [False], [False],
                 estimate_method='larger_data',
                 seed=21)
    self.assertAlmostEqual(result, expected, places=2)

    # Higher dimension.
    rho = 0.9  # fairly strongly dependent features
    [f0, f1, f2, f3], expected = self._MakeCorrelatedFeatures([1, 2, -3, 4],
                                                              rho)

    for method in ['smaller_data', 'larger_data']:
      result = _MI([f1, f2, f3], [f0], [False] * 3, [False],
                   estimate_method=method,
                   seed=21)
      self.assertAlmostEqual(result, expected, delta=2e-2)

  def testAddingIndependentFeature(self):
    # Adding an independent feature into the computation, does not alter the
    # mutual information.
    np.random.seed(23)
    r = np.random.randint(0, 8, 50000)
    s = np.random.randint(0, 3, 50000) + r
    w = np.random.randn(50000)

    for method in ['smaller_data', 'larger_data']:
      mi_rs = _MI([r], [s], [False], [False], estimate_method=method, seed=21)
      mi_rws = _MI([r, w], [s], [False] * 2, [False],
                   estimate_method=method,
                   seed=21)
      self.assertAlmostEqual(mi_rws, mi_rs, places=2)

  def testMissingValues(self):
    np.random.seed(23)
    fz = np.array([1.] * 10000)
    fx = np.random.random(10000)
    fa = np.array([1] * 5000 + [2] * 5000, dtype=float)
    fb = np.array([2.3] * 5000 + [None] * 5000)
    fc = np.array([0.] * 5000 + [10.] * 5000)

    for method in ['smaller_data', 'larger_data']:
      result = _MI([fz], [fa], [False], [False],
                   seed=23,
                   estimate_method=method)
      self.assertLess(abs(result), 1e-2)

      result = _MI([fc], [fa], [False], [False],
                   seed=23,
                   estimate_method=method)
      self.assertLess(abs(result - 1), 1e-2)

      result = _MI([fb], [fa], [False], [False],
                   seed=23,
                   estimate_method=method)
      self.assertLess(abs(result - 1), 1e-2)

      # Add an independent feature does not affect.
      result = _MI([fc, fx], [fa], [False] * 2, [False],
                   seed=23,
                   estimate_method=method)
      self.assertLess(abs(result - 1), 1e-2)

      result = _MI([fb, fx], [fa], [False] * 2, [False],
                   seed=23,
                   estimate_method=method)
      self.assertLess(abs(result - 1), 1e-2)

  def testFilterFeat(self):
    np.random.seed(3)
    fa = np.array(['cat0'] * 2000 + ['cat1'] * 2000 + ['cat2'] * 2000 +
                  ['cat3'] * 2000)  # 4 categories
    fg = np.array([1] * 2000 + [2] * 2000 + [3] * 2000 + [4] * 2000)

    filter_feat = np.array([1] * 6000 + [None] * 2000)
    filter_arr = np.array([True] * 6000 + [False] * 2000)

    for method in ['smaller_data', 'larger_data']:
      result = _MI([fg], [fa], [True], [True],
                   filter_feature=filter_arr,
                   seed=20,
                   estimate_method=method)
      self.assertAlmostEqual(result, np.log2(3), places=2)

      result = _MI([fg], [fa], [False], [True],
                   filter_feature=filter_arr,
                   seed=20,
                   estimate_method=method)
      self.assertAlmostEqual(result, np.log2(3), places=2)

      result = _MI([fg], [filter_feat], [False], [False],
                   seed=23,
                   estimate_method=method)
      self.assertAlmostEqual(result, (3 / 4) * (np.log2(4 / 3)) + 0.5, places=2)

      result = _MI([fg], [filter_feat], [False], [False],
                   filter_feature=filter_arr,
                   seed=23,
                   estimate_method=method)
      self.assertLess(abs(result), 1e-2)

  def testWeightFeat(self):
    np.random.seed(3)
    fa = np.array(['cat0'] * 2000 + ['cat1'] * 2000 + ['cat2'] * 2000 +
                  ['cat3'] * 2000)  # 4 categories
    fg = np.array([1] * 2000 + [2] * 2000 + [3] * 2000 + [4] * 2000)

    weight_feat = np.array([1] * 2000 + [0.5] * 2000 + [0.25] * 2000 +
                           [0] * 2000)

    for method in ['smaller_data', 'larger_data']:
      result = _MI([fg], [fa], [True], [True],
                   weight_feature=weight_feat,
                   seed=20,
                   estimate_method=method)
      self.assertAlmostEqual(result, 7 / 8, delta=1e-2)

      result = _MI([fg], [weight_feat], [False], [False],
                   weight_feature=weight_feat,
                   seed=23,
                   estimate_method=method)
      self.assertAlmostEqual(result, 7 / 8, delta=1e-2)

  def testAssertions(self):
    np.random.seed(23)
    fx = np.random.random(1000)
    fy = np.array([1.] * 1000)

    with self.assertRaises(AssertionError):
      _MI([], [fy], [False], [False])

    with self.assertRaises(AssertionError):
      _MI([fx], [], [False], [False])

    with self.assertRaises(AssertionError):
      _MI(fx, [fy], [False], [False])

    with self.assertRaises(AssertionError):
      _MI([fx], [fy], [False] * 2, [False])

    with self.assertRaises(AssertionError):
      _MI([fx], [fy], [False], [False], output_each='False')

  def testOutputEachSanityCheck(self):
    np.random.seed(23)
    fx = np.random.randn(1000)
    fy = np.array([1.] * 1000)
    fz = np.array([True] * 700 + [False] * 300)

    for method in ['smaller_data', 'larger_data']:
      result, each_mi = _MI([fx], [fy], [False], [False],
                            seed=3,
                            output_each=True,
                            estimate_method=method)
      self.assertLess(abs(result), 1e-2)
      self.assertLen(each_mi, 1000)
      self.assertLess(max(0, np.mean(each_mi)), 1e-2)

      result, each_mi = _MI([fx], [fy], [False], [False],
                            filter_feature=fz,
                            seed=4,
                            output_each=True,
                            estimate_method=method)
      self.assertLess(abs(result), 1e-2)
      self.assertLen(each_mi, 700)
      self.assertLess(max(0, np.mean(each_mi)), 1e-2)

  def testOutputEach(self):
    np.random.seed(97)
    n = 10000
    fx = np.random.randint(0, 8, n)

    for method in ['smaller_data', 'larger_data']:
      for categorical0, categorical1 in [(True, True), (False, True),
                                         (False, False)]:
        # Test categorical vs categorical, ordinal vs categorical, ordinal
        # vs ordinal.
        result, each_mi = _MI([fx], [fx], [categorical0], [categorical1],
                              output_each=True,
                              estimate_method=method,
                              seed=5)
        self.assertAlmostEqual(result, 3, places=1)
        self.assertLen(each_mi, n)
        self.assertAlmostEqual(np.mean(each_mi), 3, places=1)
        self.assertAlmostEqual(
            np.sum(each_mi[fx == 0]) / n, 3. / 8, places=None, delta=1e-2)

    for method in ['smaller_data', 'larger_data']:
      for categorical0, categorical1, categorical2 in [(False, False, True),
                                                       (False, True, True)]:
        result, each_mi = _MI([fx, fx], [fx], [categorical0, categorical1],
                              [categorical2],
                              output_each=True,
                              estimate_method=method,
                              seed=9)
        self.assertAlmostEqual(result, 3, places=2)
        self.assertLen(each_mi, n)
        self.assertAlmostEqual(np.mean(each_mi), 3, places=2)
        self.assertAlmostEqual(
            np.sum(each_mi[fx == 0]) / n, 3. / 8, places=None, delta=1e-2)

  def testCategorical(self):
    np.random.seed(3)
    a = np.array([b'cat0'] * 2000 + [b'cat1'] * 2000 + [b'cat2'] * 2000 +
                 [b'\xc5\x8cmura'] * 2000)  # 4 categories
    b = np.random.randn(a.size)
    c = np.arange(0.1, 100, 0.001)[:a.size] + 2 * b
    d = (
        np.random.normal(0.5, 1.0, a.size) +
        np.random.normal(-0.5, 1.0, a.size) + np.random.normal(0., 0.3, a.size))
    e = np.arange(0.1, 100, 0.001)[:a.size]
    # Build some features that repeat N times the same value sequence.
    g = np.array([i // (a.size // 8) for i in range(a.size)])
    h = np.array([b'cat%d' % (i // (a.size // 16)) for i in range(a.size)])

    for method in ['smaller_data', 'larger_data']:
      result = _MI([b], [a], [False], [True],
                   k=6,
                   estimate_method=method,
                   seed=20)
      self.assertLess(abs(result), 2e-2)

      result = _MI([c], [a], [False], [True],
                   k=6,
                   estimate_method=method,
                   seed=20)
      self.assertAlmostEqual(result, 0.565, delta=1e+2)

      result = _MI([d], [a], [False], [True],
                   k=6,
                   estimate_method=method,
                   seed=20)
      self.assertLess(abs(result), 1e-2)

      result = _MI([e], [h], [False], [True],
                   k=6,
                   estimate_method=method,
                   seed=20)
      self.assertAlmostEqual(result, 4, delta=1e+2)

      result = _MI([g], [h], [False], [True],
                   k=6,
                   estimate_method=method,
                   seed=20)
      self.assertAlmostEqual(result, 3, delta=1e+2)

      result = _MI([a, b], [b, a], [True, False], [False, True],
                   estimate_method=method,
                   seed=20)
      self.assertAlmostEqual(result, 13.15, delta=1e+2)

  def testCategoricalOrdinal(self):
    np.random.seed(3)
    # Feature B has PDF 3/4 in [0, 1] vs 1/4 in [1, 2], and differential entropy
    #   H(B) = - 3/4 * log(3/4) - 1/4 * log(1/4)
    # while, given A, it has conditional entropy
    #   H(B | A) = 1/2 * H(B | A == 0) + 1/2 * H(B | A == 1)
    #   H(B | A) = 1/2 * 0. - 1/2 * log(1/2) = - 1/2 * log(1/2)
    # hence their mutual information is
    #   I(A, B) = H(B) - H(B | A) = - 3/4 * log(3/4)
    # using whatever log base we're using, in this case base 2.
    a = np.array([i % 2 for i in range(1000)])
    b = np.array([np.random.random() * (1. + i % 2) for i in range(1000)])
    filt = np.array([True if i % 2 else False for i in range(1000)])
    for method in ['smaller_data', 'larger_data']:
      self.assertAlmostEqual(
          -0.75 * np.log2(0.75),
          _MI([a], [b], [True], [False], estimate_method=method, seed=20),
          delta=2e-2)
      # If we filter out 1 of the 2 A labels however, no information is left.
      self.assertEqual(
          0.,
          _MI([a], [b], [True], [False],
              estimate_method=method,
              seed=20,
              filter_feature=filt))

  def testAdjustedMutualInformation(self):
    np.random.seed(11)
    f0 = np.random.randint(0, 10000, 10000)
    label = np.array([0, 1] * 5000)

    result = mutual_information_util.mutual_information([f0], [label], [True],
                                                        [True],
                                                        seed=11)
    adjusted_result = _AMI([f0], [label], [True], [True], seed=11)
    self.assertAlmostEqual(result, 0.625, delta=2e-2)
    self.assertAlmostEqual(adjusted_result, 0.0, delta=2e-2)

  def testMergeCategorical(self):
    actual = mutual_information_util._merge_categorical([
        np.array(['a', 'b', 'c']),
        np.array(['1', '2', '3']),
        np.array(['alpha', 'beta', 'gamma'])
    ])
    self.assertTrue(
        np.array_equal(
            np.array([b'a:1:alpha', b'b:2:beta', b'c:3:gamma']), actual))

  def testEntropyD(self):
    discrete_f = np.array(['foo', 'bar', 'baz', 'foo'])
    entropy, each = mutual_information_util._entropy_discrete(
        discrete_f, np.ones_like(discrete_f, dtype=float))
    expected_entropy = -(np.log2(0.5) * 0.5 + np.log2(0.25) * 0.25 * 2)
    expected_each = np.array(
        [-np.log2(0.5), -np.log2(0.25), -np.log2(0.25), -np.log2(0.5)])
    self.assertTrue(np.allclose(expected_entropy, entropy, atol=1e-5))
    self.assertTrue(np.allclose(expected_each, each, atol=1e-5))

  def testReplaceNoneC(self):
    arr = np.array([1.0, 2.0, np.nan])
    expected = np.array(
        [1.0, 2.0, 2 * 2.0 - 1.0 + mutual_information_util._NONE_NUM])
    actual = mutual_information_util._replace_none_categorical(arr)
    self.assertTrue(np.array_equal(expected, actual))

  def testUnitVarianceScale(self):
    arr = np.array([1.0, 2.0, np.nan])
    actual = mutual_information_util._unit_variance_scale(arr)
    stdev = np.std([1.0, 2.0], ddof=1)
    self.assertTrue(
        np.allclose(
            np.array([(1.0 - 1.5) / stdev, (2 - 1.5) / stdev]),
            actual[~np.isnan(actual)],
            atol=1e-5))

  def testUnitVarianceScale_UniformValues(self):
    arr = np.array([1.0, 1.0, np.nan])
    expected = np.array([0.0, 0.0, np.nan])
    actual = mutual_information_util._unit_variance_scale(arr)
    np.testing.assert_equal(actual[np.isnan(actual)],
                            expected[np.isnan(expected)])
    self.assertTrue(
        np.allclose(
            expected[~np.isnan(expected)], actual[~np.isnan(actual)],
            atol=1e-5))

  def testFeatureToNumpyArray(self):
    feat = np.array([1.0, 2.0, None])
    expected = np.array([1.0, 2.0, np.nan])
    actual = mutual_information_util._fill_missing_values(feat, False)
    np.testing.assert_equal(actual[np.isnan(actual)],
                            expected[np.isnan(expected)])
    np.testing.assert_equal(expected, actual)

    feat = np.array([b'a', b'b', None])
    expected = np.array([b'a', b'b', np.nan], dtype=object)
    actual = mutual_information_util._fill_missing_values(feat, True)
    self.assertEqual([
        i for i, v in enumerate(actual) if isinstance(v, float) and np.isnan(v)
    ], [
        i for i, v in enumerate(expected)
        if isinstance(v, float) and np.isnan(v)
    ])
    self.assertEqual([v for v in actual if not isinstance(v, float)],
                     [v for v in expected if not isinstance(v, float)])

  def testDiscreteLabelsAppearingExactlyOnce(self):
    feat0 = np.arange(10)
    feat1 = np.arange(10, 20).astype(int)
    with self.assertRaisesRegex(
        ValueError, '.* tuples .* discrete features .* are all unique.*'):
      mutual_information_util._mi_for_arrays([feat0], [], [], [feat1],
                                             np.ones_like(feat1))


if __name__ == '__main__':
  absltest.main()
