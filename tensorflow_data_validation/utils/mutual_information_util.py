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

"""Module for measuring mutual information using kNN method.

Using binning method for computing mutual information of ordinal features is
not applicable in real use cases due to the bad accuracy. Instead, this module
implements:

(1) the algorithm in PRE paper 10.1103/PhysRevE.69.066138 (See the paper online
at http://arxiv.org/abs/cond-mat/0305641v1) for estimating the mutual
information between ordinal features.

(2) the algorithm in PLOS paper PLoS ONE 9(2): e87357 (
http://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0087357) for
estimating the mutual information between ordinal features and categorical
features. Besides, this module also handles missing values and weighted samples.

For each algorithm there are two estimate methods '1' and '2' as described in
the PRE paper. Scikit-learn(dev) and several other Python open source
implementations only provide the method '1' with feature dimension 2 and without
handling missing values or weights.

The two methods should usually produce similar results. Method '1' has smaller
statistical errors, while method '2' has smaller systematic errors. So method
'1' is more suitable for low dimensional small data sets, while method '2' is
more preferred for high dimensional larger data sets. The default method is '2'.

In this implementation, we use the more understandable names 'smaller_data' and
'larger_data' to represent the methods '1' and '2'.

The results are insensitive to the value of k. In practice, usually people use
k = 3. Larger values are also fine.

The major problem with kNN estimation method is that it would fail if the
features have lots of samples have the same value or very close values. The PRE
paper has two suggestions that usually work:

(a) Add a tiny noise. This not only breaks the data degeneracy but also speeds
up the computation a lot. This is because it decreases the number of neighbors
by breaking the data degeneracy. And this trick does not affect the quality of
the result. This functionality is controlled by 'seed' and '_NOISE_AMPLITUDE'
constant.

(b) Reparametrize the features, for example, M.Log(A) can re-distribute data
within a small range into a wider range. Any homeomorphism transformation of a
feature does not alter the mutual information, so this does not affect the
result either. Of course any homeomorphism is fine. This is helpful especially
when the feature value distributed is highly skewed. Strictly speaking, this is
not required, but it usually decreases the error. Re-scaling the features to
have unit variance also helps decreasing errors.
"""

# NOTE: In the code, some variable names start with "c" and some with "d"; this
# is because the key distinction between categorical vs ordinal features is
# closely related to, hence vaguely conflated with, "continuous" vs "discrete".


import functools
import itertools
import math
from typing import Any, List, Optional, Tuple, Union
import uuid

import numpy as np
import pandas as pd
import scipy.special
import sklearn.neighbors


# For categorical features, we will use this unique string to represent missing
# values and handle it as if it was a normal value.
_NONE_STR = str(uuid.uuid4()).encode()

# For ordinal features, we will use Max(feat) + Max(feat) - Min(feat)
# + _NONE_NUM to represent missing values and handle it as if it was a normal
# value.
_NONE_NUM = 10.

# When considering the k nearest neighbors, it could cause problems if two
# neighbors have the same distance. Do we want to include one of them or both of
# them? So we use a tiny noise to break the tie, which does not affect the
# mutual information value.
_NOISE_AMPLITUDE = 1e-10


def mutual_information(
    feature_list0: List[np.ndarray],
    feature_list1: List[np.ndarray],
    is_categorical_list0: List[bool],
    is_categorical_list1: List[bool],
    k: int = 3,
    estimate_method: str = 'larger_data',
    weight_feature: Optional[np.ndarray] = None,
    filter_feature: Optional[np.ndarray] = None,
    output_each: bool = False,
    seed: Optional[int] = None) -> Union[float, Tuple[float, np.ndarray]]:
  """Computes MI between two lists of features (numpy arrays).

  The mutual information value is scaled by log(2) in the end so that the unit
  is bit.

  The paper (1) in the module doc string gives the method for computing MI
  between two lists of ordinal features. The paper (2) provides the method
  for computing MI between a list of ordinal features and a list of categorical
  features. For the general case, suppose we have ordinal feature set C0, C1,
  and categorical feature set D0, D1. Then we can derive

  I({C0,D0};{C1,D1}) = I({C0,C1};{D0,D1}) + I(C0;C1) + I(D0;D1) - I(C0;D0)
                       - I(C1;D1),

  where the right hand side terms can all be computed by using the methods in
  the two papers.

  Args:
    feature_list0: (list(np.ndarray)) A list of features.
    feature_list1: (list(np.ndarray)) A list of features.
    is_categorical_list0: (list(bool)) Whether the first list of features are
      categorical or not.
    is_categorical_list1: (list(bool)) Whether the second list of features are
      categorical or not.
    k: (int) The number of nearest neighbors. It has to be an integer no less
      than 3.
    estimate_method: (str) 'smaller_data' or 'larger_data' estimator in the
      above paper.
    weight_feature: (np.ndarray) A feature that contains weights for each
      sample.
    filter_feature: (np.ndarray) A feature that is used as the filter to drop
      all data where this filter has missing values. By default, it is None and
      no filtering is done.
    output_each: (bool) Whether to output the contribution from each individual
      sample. The output values are not scaled by the number of samples.
    seed: (int) Random seed for the tiny noise.

  Returns:
    (float | (float, np.ndarray)) The mutual information between the features in
        feature_list0 and feature_list1. If output_each is True, an np array of
        the contributions from all samples is also output, whose mean is equal
        to the mutual information.
  """
  _validate_args(feature_list0, feature_list1, is_categorical_list0,
                 is_categorical_list1, k, estimate_method, weight_feature,
                 filter_feature, output_each, seed)

  cf_list0, cf_list1, df_list0, df_list1, weights = _feature_list_to_numpy_arrays(
      feature_list0, feature_list1, is_categorical_list0, is_categorical_list1,
      weight_feature, filter_feature)

  # Try to reuse these data in later computations to avoid converting Feature to
  # numpy array multiple times.
  final_mi, each = _mi_for_arrays(cf_list0, cf_list1, df_list0, df_list1,
                                  weights, k, estimate_method, seed)
  if output_each:
    return final_mi, each
  return final_mi


def adjusted_mutual_information(
    feature_list0: List[np.ndarray],
    feature_list1: List[np.ndarray],
    is_categorical_list0: List[bool],
    is_categorical_list1: List[bool],
    k: int = 3,
    estimate_method: str = 'larger_data',
    weight_feature: Optional[np.ndarray] = None,
    filter_feature: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> float:
  """Computes adjusted MI between two lists of features.

  Args:
    feature_list0: (list(np.ndarray)) a list of features represented as numpy
      arrays.
    feature_list1: (list(np.ndarray)) a list of features represented as numpy
      arrays.
    is_categorical_list0: (list(bool)) Whether the first list of features are
      categorical or not.
    is_categorical_list1: (list(bool)) Whether the second list of features are
      categorical or not.
    k: (int) The number of nearest neighbors. It has to be an integer no less
      than 3.
    estimate_method: (str) 'smaller_data' or 'larger_data' estimator in the
      above paper.
    weight_feature: (np.ndarray) numpy array that are weights for each example.
    filter_feature: (np.ndarray) numpy array that is used as the filter to drop
      all data where this has missing values. By default, it is None and no
      filtering is done.
    seed: (int) the numpy random seed.

  Returns:
    The adjusted mutual information between the features in feature_list0 and
    feature_list1.
  """
  _validate_args(feature_list0, feature_list1, is_categorical_list0,
                 is_categorical_list1, k, estimate_method, weight_feature,
                 filter_feature, False, seed)

  cf_list0, cf_list1, df_list0, df_list1, weights = _feature_list_to_numpy_arrays(
      feature_list0, feature_list1, is_categorical_list0, is_categorical_list1,
      weight_feature, filter_feature)

  return _adjusted_mi_for_arrays(cf_list0, cf_list1, df_list0, df_list1,
                                 weights, k, estimate_method, seed)


def _mi_for_arrays(c_arrs0: List[np.ndarray],
                   c_arrs1: List[np.ndarray],
                   d_arrs0: List[np.ndarray],
                   d_arrs1: List[np.ndarray],
                   weights: Optional[np.ndarray] = None,
                   k: int = 3,
                   estimate_method: str = 'larger_data',
                   seed: Optional[int] = None) -> Tuple[float, np.ndarray]:
  """Computes MI for a list of np.ndarrays."""
  assert (bool(c_arrs0 + d_arrs0) and
          bool(c_arrs1 + d_arrs1)), 'Both sides are expected to be nonempty.'
  fs = list(itertools.chain(c_arrs0, c_arrs1, d_arrs0, d_arrs1))
  for other_f in fs[1:]:
    assert len(fs[0]) == len(other_f)

  np.random.seed(seed)

  # Scale ordinal features, and replace missing values in all features.
  c_arrs0 = [
      _replace_none_categorical(_unit_variance_scale(f)) for f in c_arrs0
  ]
  c_arrs1 = [
      _replace_none_categorical(_unit_variance_scale(f)) for f in c_arrs1
  ]
  d_arrs0 = [_to_dense_discrete_array(f) for f in d_arrs0]
  d_arrs1 = [_to_dense_discrete_array(f) for f in d_arrs1]

  arr0 = _to_noisy_numpy_array(c_arrs0)
  arr1 = _to_noisy_numpy_array(c_arrs1)
  df0 = _merge_categorical(d_arrs0)
  df1 = _merge_categorical(d_arrs1)

  if weights is None:
    weights = np.ones_like(fs[0], dtype=float)

  if (arr0 is None and arr1 is None) or (df0 is None and df1 is None):
    mi_c01_d01, each_c01_d01 = 0., 0.
  else:
    arr = np.hstack(([] if arr0 is None else [arr0]) +
                    ([] if arr1 is None else [arr1]))
    df = _merge_categorical(([] if df0 is None else [df0]) +
                            ([] if df1 is None else [df1]))
    mi_c01_d01, each_c01_d01 = _mi_high_dim_cd(arr, df, k, estimate_method,
                                               weights)

  if arr0 is None or arr1 is None:
    mi_c0_c1, each_c0_c1 = 0., 0.
  else:
    mi_c0_c1, each_c0_c1 = _mi_high_dim_cc(arr0, arr1, k, estimate_method,
                                           weights)

  if df0 is None or df1 is None:
    mi_d0_d1, each_d0_d1 = 0., 0.
  else:
    mi_d0_d1, each_d0_d1 = _mi_high_dim_dd(df0, df1, weights)

  if arr0 is None or df0 is None:
    mi_c0_d0, each_c0_d0 = 0., 0.
  else:
    mi_c0_d0, each_c0_d0 = _mi_high_dim_cd(arr0, df0, k, estimate_method,
                                           weights)

  if arr1 is None or df1 is None:
    mi_c1_d1, each_c1_d1 = 0., 0.
  else:
    mi_c1_d1, each_c1_d1 = _mi_high_dim_cd(arr1, df1, k, estimate_method,
                                           weights)

  final_mi = max(0., mi_c01_d01 + mi_c0_c1 + mi_d0_d1 - mi_c0_d0 - mi_c1_d1)
  each = each_c01_d01 + each_c0_c1 + each_d0_d1 - each_c0_d0 - each_c1_d1
  assert isinstance(each, np.ndarray)

  return final_mi, each


def _adjusted_mi_for_arrays(
    c_arrs0: List[np.ndarray],
    c_arrs1: List[np.ndarray],
    d_arrs0: List[np.ndarray],
    d_arrs1: List[np.ndarray],
    weights: Optional[np.ndarray] = None,
    k: int = 3,
    estimate_method: str = 'larger_data',
    seed: Optional[int] = None,
) -> float:
  """Computes AdjustedMutualInformation for given np.ndarrays.

  Args:
    c_arrs0: Continuous arrays for side 0.
    c_arrs1: Continuous arrays for side 1.
    d_arrs0: Discrete arrays for side 0.
    d_arrs1: Discrete arrays for side 1.
    weights: Weights for data points.
    k: The number of nearest neighbors to check when computing MI.
    estimate_method: Underlying estimate method for computing MI.
    seed: The seed for RNGs.

  Returns:
    AMI
  """
  if seed is not None:
    np.random.seed(seed)

  # Always set `output_each` to be False.
  seed1 = None if seed is None else np.random.randint(0, 1000)
  mi, _ = _mi_for_arrays(c_arrs0, c_arrs1, d_arrs0, d_arrs1, weights, k,
                         estimate_method, seed1)

  # We use the same seed to shuffle several features together.
  shuffle_seed = np.random.randint(0, 1000)  # a fixed seed for shuffling
  array_length = next(itertools.chain(c_arrs0, c_arrs1, d_arrs0, d_arrs1)).size
  np.random.seed(shuffle_seed)
  shuffled_index = np.random.permutation(array_length)

  shuffled_c_arrs0 = [a[shuffled_index] for a in c_arrs0]
  shuffled_d_arrs0 = [a[shuffled_index] for a in d_arrs0]

  seed2 = None if seed is None else np.random.randint(0, 1000)
  mi_shuffled, _ = _mi_for_arrays(shuffled_c_arrs0, c_arrs1, shuffled_d_arrs0,
                                  d_arrs1, weights, k, estimate_method, seed2)

  return max(mi - mi_shuffled, 0.0)


def _to_dense_discrete_array(f: np.ndarray) -> np.ndarray:
  ret = f.astype(bytes)
  ret[pd.isnull(f)] = _NONE_STR
  return ret


def _replace_none_categorical(f: np.ndarray) -> np.ndarray:
  """Replaces missing values in a ordinal feature."""
  if np.all(np.isnan(f)):
    return np.full_like(f, _NONE_NUM)
  # Replace the missing value with a large enough float value so that when
  # looking for k nearest neighbors, samples with missing values are treated
  # separately (only samples with the same missing values are taken into account
  # for nearest neighbors).
  return np.nan_to_num(
      f, copy=True, nan=2 * np.nanmax(f) - np.nanmin(f) + _NONE_NUM)


def _unit_variance_scale(f: np.ndarray) -> np.ndarray:
  """Rescales a feature to have a unit variance."""
  f_nan_max = np.nanmax(f)
  f_nan_min = np.nanmin(f)
  if np.isnan(f_nan_max) or np.isnan(f_nan_min):
    raise ValueError('Continuous feature all missing.')
  if f_nan_max == f_nan_min:
    ret = np.full_like(f, np.nan, dtype=float)
    ret[~np.isnan(f)] = 0
    return ret
  return (f - np.nanmean(f)) / np.nanstd(f, ddof=1)


def _merge_categorical(discrete_fs: List[np.ndarray]) -> Any:
  """Merges a list of categorical features into a single categorical feature."""
  if not discrete_fs:
    return None
  operand_list = []
  for i in range(2 * len(discrete_fs) - 1):
    if i % 2 == 0:
      operand_list.append(discrete_fs[i // 2].astype(bytes))
    else:
      operand_list.append(b':')  # use ':' to join values
  return functools.reduce(np.char.add, operand_list)


def _entropy_discrete(discrete_f: np.ndarray,
                      weight_f: np.ndarray) -> Tuple[float, np.ndarray]:
  """Computes the entropy of a list of categorical features with weights."""
  _, inverse_idx, unique_counts = np.unique(
      discrete_f, return_inverse=True, return_counts=True)
  group_counts = unique_counts[inverse_idx]
  each = -np.log2(group_counts / discrete_f.size) * weight_f
  return np.mean(each), each


def _assert_feature_list(feature_list: List[np.ndarray],
                         list_name: str) -> None:
  """Validates the contents of feature_list arg for `mutual_information`."""
  for f in feature_list:
    if f.dtype == float:
      mask = (f == float('inf')) | (f == float('-inf'))
      assert np.sum(mask) == 0, (
          'Feature list: %s in list %s contains infinite values, which '
          'currently are not supported.' % (f, list_name))


def _validate_args(
    feature_list0: List[np.ndarray],
    feature_list1: List[np.ndarray],
    is_categorical_list0: List[bool],
    is_categorical_list1: List[bool],
    k: int,
    estimate_method: str,
    weight_feature: np.ndarray,
    filter_feature: np.ndarray,
    output_each: bool,
    seed: Optional[int]) -> None:
  """Validates the arguments of the function `mutual_information`."""

  assert len(set(len(f) for f in feature_list0 + feature_list1)) == 1, (
      'The features have different number of items.')

  assert len(is_categorical_list0) == len(feature_list0), (
      'is_categorical_list0 is not the same length as feature_list0.')
  assert len(is_categorical_list1) == len(feature_list1), (
      'is_categorical_list1 is not the same length as feature_list1.')

  assert isinstance(k, int) and k >= 3, 'k has to be an integer no less than 3.'

  assert estimate_method in ['smaller_data', 'larger_data']

  def assert_feature(f, f_name):
    assert (f is None or isinstance(f, np.ndarray) and
            len(f) == len(feature_list0[0])), (
                '%s must be None or a feature with the same item number.' %
                f_name)

  assert_feature(weight_feature, 'weight_feature')
  assert_feature(filter_feature, 'filter_feature')

  assert isinstance(output_each, bool)
  assert seed is None or isinstance(seed, int) and seed > 0


def _fill_missing_values(f: np.ndarray, is_categorical: bool) -> np.ndarray:
  """Fills `f` with `np.nan` for missing values.

  Missing values are represented with `np.nan`, regardless of the dtype of the
  returned np.ndarray. All continuous features (i.e. is_categorical == False)
  are cast to float.

  E.g.
    np.array([1, 2, None]) -> np.array([1.0, 2.0, nan], dtype=float)
    np.array(['a', None, None]) -> np.array(['a', nan, nan], dtype=object)

  Args:
    f: np.ndarray.
    is_categorical: bool.

  Returns:
    np.ndarray.
  """
  if is_categorical:
    f = f.astype(object)
    f[pd.isnull(f)] = np.nan
    return f
  else:
    # Converting to np.float64 is necessary for getting smaller errors.
    return f.astype(float)


def _feature_list_to_numpy_arrays(
    feature_list0: List[np.ndarray], feature_list1: List[np.ndarray],
    is_categorical_list0: List[bool], is_categorical_list1: List[bool],
    weight_feature: Optional[np.ndarray], filter_feature: Optional[np.ndarray]
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray],
           List[np.ndarray], np.ndarray]:
  """Converts feature lists into np.ndarray lists for MI computation."""
  n_samples = len(feature_list0[0])

  if weight_feature is None:  # the default weight is constant 1
    weights = np.ones(n_samples).astype(float)
  else:
    weights = weight_feature.astype(float)

  # We will handle ordinal and categorical features differently.
  def select_features(feature_list, is_categorical_list, keep_fn):
    return [
        _fill_missing_values(f, is_categorical)
        for f, is_categorical in zip(feature_list, is_categorical_list)
        if keep_fn(is_categorical)
    ]

  # Select ordinal features and categorical features.
  cf_list0 = select_features(feature_list0, is_categorical_list0,
                             lambda a: not a)
  cf_list1 = select_features(feature_list1, is_categorical_list1,
                             lambda a: not a)
  df_list0 = select_features(feature_list0, is_categorical_list0, lambda a: a)
  df_list1 = select_features(feature_list1, is_categorical_list1, lambda a: a)

  # Ignore those samples whose the filter_feature is missing.
  if filter_feature is not None:
    cf_list0 = [f[filter_feature] for f in cf_list0]
    df_list0 = [f[filter_feature] for f in df_list0]
    cf_list1 = [f[filter_feature] for f in cf_list1]
    df_list1 = [f[filter_feature] for f in df_list1]
    weights = weights[filter_feature]
  return cf_list0, cf_list1, df_list0, df_list1, weights


def _to_noisy_numpy_array(cf_list: List[np.ndarray]) -> Optional[np.ndarray]:
  """Adds a tiny noise onto ordinal features."""
  # In order to use double precision computation to get smaller errors, we add
  # noise after the features have been converted to numpy arrays.
  if not cf_list:
    return None

  arr = np.hstack([l.reshape((-1, 1)) for l in cf_list])
  # This may add a noise that is too big for features with very small mean. So
  # far it works fine, but should change it if it poses a problem.
  means = np.maximum(1, np.mean(np.abs(arr), axis=0))
  arr += (_NOISE_AMPLITUDE * means * np.random.randn(*arr.shape))
  return arr


def _process_high_dim(arr: np.ndarray, radius: int, estimate_method: str,
                      weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Processes high dimensional feature in the same way as 1-d feature."""
  kd_tree = sklearn.neighbors.KDTree(arr, metric='chebyshev')
  radius_ns = kd_tree.query_radius(X=arr, r=radius, count_only=True)

  if estimate_method == 'smaller_data':
    each = -scipy.special.digamma(radius_ns) * weights
  elif estimate_method == 'larger_data':
    each = -scipy.special.digamma(radius_ns - 1) * weights
  return np.sum(each), each


def _mi_high_dim_cc(arr0: np.ndarray, arr1: np.ndarray, k: int,
                    estimate_method: str,
                    weights: np.ndarray) -> Tuple[float, np.ndarray]:
  """Computes high dimensional MI for ordinal features."""
  arr = np.hstack([arr0, arr1])
  m0 = arr0.shape[1]
  n_samples, _ = arr.shape

  nn = sklearn.neighbors.NearestNeighbors(
      metric='chebyshev', n_neighbors=k, n_jobs=1)
  nn.fit(arr)
  k_neighbors = nn.kneighbors()

  if estimate_method == 'smaller_data':
    # Use one radius for all features. Exclude the point on the boundary by
    # taking a radius slightly smaller than the distance to the k-th nearest
    # neighbor.
    r = np.nextafter(k_neighbors[0][:, -1], 0).reshape((-1, 1))
    radius = np.hstack([r, r])
  elif estimate_method == 'larger_data':
    # Treat arr0 and arr1 as two high dimensional features and each of them uses
    # its own projection of the radius. The idea is to look at the k nearest
    # neighbors and find the radius (largest distance) in the two sub-spaces
    # separately. The following code does this for chebyshev distance metric.
    ind = k_neighbors[1][:, 0]
    r = np.fabs(arr - arr[ind])
    for i in range(1, k_neighbors[1].shape[1]):
      ind = k_neighbors[1][:, i]
      r = np.maximum(r, np.fabs(arr - arr[ind]))
    r0 = np.max(r[:, :m0], axis=1).reshape((-1, 1))
    r1 = np.max(r[:, m0:], axis=1).reshape((-1, 1))
    radius = np.hstack([r0, r1])

  mi0, each0 = _process_high_dim(arr0, radius[:, 0], estimate_method, weights)
  mi1, each1 = _process_high_dim(arr1, radius[:, 1], estimate_method, weights)
  mi = (mi0 + mi1) / float(n_samples)

  if estimate_method == 'smaller_data':
    extra = (scipy.special.digamma(k) +
             scipy.special.digamma(n_samples)) * weights
  elif estimate_method == 'larger_data':
    extra = (scipy.special.digamma(k) + scipy.special.digamma(n_samples) -
             1. / k) * weights
  mi += np.mean(extra)
  each = each0 + each1 + extra

  final_mi = max(0., mi / math.log(2))
  return final_mi, each / math.log(2)


def _mi_high_dim_cd(arr: np.ndarray, arr_d: np.ndarray, k: int,
                    estimate_method: str,
                    weights: np.ndarray) -> Tuple[float, np.ndarray]:
  """Computes high dimensional MI between ordinal and categorical features."""
  n_samples = arr_d.size
  radius = np.empty(n_samples)
  label_counts = np.empty(n_samples)
  k_all = np.empty(n_samples)

  nn = sklearn.neighbors.NearestNeighbors(
      metric='chebyshev', n_neighbors=k, n_jobs=1)
  each = np.zeros(n_samples)
  for label in np.unique(arr_d):
    mask = arr_d == label
    count = np.sum(mask)
    if count > 1:
      cur_k = min(k, count - 1)

      nn.set_params(n_neighbors=cur_k)
      nn.fit(arr[mask])
      k_neighbors = nn.kneighbors()
      if estimate_method == 'smaller_data':
        # When we count the number of points that fall in the sphere of this
        # radius in each of the two sub feature spaces, we need to exclude the
        # points on the boundary by taking a radius slightly smaller than the
        # distance to the k-th nearest neighbor.
        radius[mask] = np.nextafter(k_neighbors[0][:, -1], 0)
      elif estimate_method == 'larger_data':
        radius[mask] = k_neighbors[0][:, -1]
      k_all[mask] = cur_k
    label_counts[mask] = count

  # Ignore the labels that contain only one data point.
  mask = label_counts > 1
  if not np.any(mask):
    raise ValueError(
        'The tuples defined by discrete features (of either side) are all '
        'unique.')

  n_samples = np.sum(mask)
  label_counts = label_counts[mask]
  k_all = k_all[mask]
  arr = arr[mask]
  radius = radius[mask]
  weights = weights[mask]

  mi, mi_each = _process_high_dim(arr, radius, estimate_method, weights)
  mi /= n_samples

  extra = (scipy.special.digamma(n_samples) + scipy.special.digamma(k_all) -
           scipy.special.digamma(label_counts)) * weights
  mi += np.mean(extra)
  each[mask] += mi_each + extra

  final_mi = max(0., mi / math.log(2))
  return final_mi, each / math.log(2)


def _mi_high_dim_dd(df0: np.ndarray, df1: np.ndarray,
                    weight_f: np.ndarray) -> Tuple[float, np.ndarray]:
  """Computes high dimensional MI for categorical features."""
  mi0, each0 = _entropy_discrete(df0, weight_f)
  mi1, each1 = _entropy_discrete(df1, weight_f)
  mi01, each01 = _entropy_discrete(_merge_categorical([df0, df1]), weight_f)
  mi = mi0 + mi1 - mi01
  final_mi = max(0., mi)
  return final_mi, each0 + each1 - each01
