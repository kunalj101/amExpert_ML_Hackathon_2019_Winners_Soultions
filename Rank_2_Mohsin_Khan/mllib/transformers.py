"""All the transformers go here."""

from abc import abstractmethod
from collections import Counter, defaultdict
import itertools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import cosine
from tqdm import tqdm

from mllib.utils import load_npy


def _convert_to_2d_array(X):
    X = np.array(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


class BaseTransformer(BaseEstimator, TransformerMixin):
    """Base interface for transformer."""

    def fit(self, X, y=None):
        """Learn something from data."""
        return self

    @abstractmethod
    def _transform(self, X):
        pass

    def transform(self, X, y=None):
        """Transform data."""
        return self._transform(X)


class ArrayTransformer(BaseTransformer):
    """Transformer to be used for returnng 2d arrays."""

    def transform(self, X):
        """Transform data and return 2d array."""
        Xt = self._transform(X)
        return _convert_to_2d_array(Xt)


class SelectCols(BaseTransformer):
    """Select column of a dataframe."""

    def __init__(self, cols):
        """Initialie columns to be selected."""
        self.cols = cols

    def _transform(self, X, y=None):
        return X[self.cols]


class GroupCatCatNUnique(ArrayTransformer):
    """Get no. of unique values for a column after grouping by another column."""

    def __init__(self, grouper_col, value_col):
        """Select group column and column to be grouped."""
        self.grouper_col = grouper_col
        self.value_col = value_col

    def _transform(self, X, y=None):
        mapping = X.groupby(self.grouper_col)[self.value_col].nunique()
        return X[self.grouper_col].map(mapping)


class FunctionTransfomer(ArrayTransformer):
    """Apply a func on arrays which returns back arrays."""

    def __init__(self, func):
        """Initialize function to be used."""
        self.func = func

    def _transform(self, X):
        return self.func(X)


class CatCountEncoder(ArrayTransformer):
    """Map counts of a value in column usin training data."""

    def fit(self, X, y=None):
        """Learn count mapping."""
        self.cat_counts = Counter(X)
        return self

    def _transform(self, X, y=None):
        return np.array([self.cat_counts[x] for x in tqdm(X)])


class CatCatUniqueCount(ArrayTransformer):
    """Learn nuique value of a column w.r.t other from training data."""

    def __init__(self, grouper_col, value_col):
        """Get grouper and to be grouped columns."""
        self.grouper_col = grouper_col
        self.value_col = value_col
        self.cat_unq_count = defaultdict(Counter)

    def fit(self, X, y=None):
        """Learn mapping."""
        grouped = X.groupby(self.grouper_col)
        self.cat_unq_count = grouped[self.value_col].nunique().to_dict()
        return self

    def _transform(self, X):
        return X[self.grouper_col].map(self.cat_unq_count)


class GroupCatCountEncoder(ArrayTransformer):
    """Map count of values for a column using only transformation data."""

    def __init__(self, cols):
        self.cols = cols

    def _transform(self, X):
        Xs = X[self.cols].to_records(index=False).tolist()
        counts = Counter(Xs)
        return np.array([counts[x] for x in Xs])


class AllCountEncoder(ArrayTransformer):
    """Count over provided data source."""

    def __init__(self, cols, data):
        self.cols = cols
        self.data = data
        self.counts = self._get_counts()

    def _transform(self, X):
        Xs = X[self.cols].to_records(index=False).tolist()
        return [self.counts[x] for x in Xs]

    def _get_counts(self):
        dr = self.data[self.cols].to_records(index=False).tolist()
        return Counter(dr)


class ExpandingTransformer(ArrayTransformer):
    """Expanding operations on hostorical artifcats."""

    FILL_VALUE = -1
    N = None

    def __init__(self, date_col, user_col, key_col, hist_artifact):
        """Initialization."""
        self.date_col = date_col
        self.user_col = user_col
        self.key_col = key_col
        self.cols = [self.date_col, self.user_col]
        self.hist_artifact = hist_artifact

    @abstractmethod
    def _reduce_func(self, arr):
        pass

    def _transform(self, X):
        Xt = []
        X = X[self.cols]
        records = X.to_records(index=False)
        col_idx = {col: i for i, col in enumerate(X.columns)}
        for row in tqdm(records):
            user = row[col_idx[self.user_col]]
            date = row[col_idx[self.date_col]]
            value = self.hist_artifact.user_date_value(
                user, date, self.key_col, reduce_func=self._reduce_func, n=self.N
            )
            if not value:
                value = self.FILL_VALUE
            Xt.append(value)
        return Xt


class DateDiff(ArrayTransformer):
    """Expanding operations on hostorical artifcats."""

    FILL_VALUE = -1

    def __init__(self, date_col, user_col, hist_artifact):
        """Initialization."""
        self.date_col = date_col
        self.user_col = user_col
        self.cols = [self.date_col, self.user_col]
        self.hist_artifact = hist_artifact

    def _transform(self, X):
        Xt = []
        X = X[self.cols]
        records = X.to_records(index=False)
        col_idx = {col: i for i, col in enumerate(X.columns)}
        for row in tqdm(records):
            user = row[col_idx[self.user_col]]
            date = row[col_idx[self.date_col]]
            date = int(date)
            prev_date_index = self.hist_artifact._get_date_index(user, date)
            if prev_date_index <= 0:
                value = self.FILL_VALUE
            else:
                time_arr = self.hist_artifact.user_time_arr.get(user, None)
                if not time_arr:
                    value = self.FILL_VALUE
                else:
                    value = (date - time_arr[prev_date_index - 1]) / 10 ** 9
            Xt.append(value)
        return Xt


class ExpandingCount(ExpandingTransformer):
    """Expanding count based on historical data."""

    def _reduce_func(self, arr):
        return len(arr)


class ExpandingSum(ExpandingTransformer):
    """Expanding count based on historical data."""

    def _reduce_func(self, arr):
        return sum(arr)


class ExpandingMax(ExpandingTransformer):
    """Expanding count based on historical data."""

    def _reduce_func(self, arr):
        return max(arr)


class ExpandingMean(ExpandingTransformer):
    """Expanding count based on historical data."""

    def _reduce_func(self, arr):
        if len(arr) == 0:
            return -1
        if isinstance(arr[0], list):
            arr = list(itertools.chain.from_iterable(arr))
        return np.sum(arr) / len(arr)


class ExpandingMedian(ExpandingTransformer):
    """Expanding count based on historical data."""

    def _reduce_func(self, arr):
        return np.median(arr)


class ZeroPct(ExpandingTransformer):
    """Expanding count based on historical data."""

    def _reduce_func(self, arr):
        arr = list(itertools.chain.from_iterable(arr))
        return np.sum(arr == 0) / len(arr)


class SetMean(ExpandingTransformer):
    """Expanding mean after merging data based on historical data."""

    def _reduce_func(self, arr):
        arr = list(itertools.chain.from_iterable(arr))
        return np.mean(arr)


class HistVectorMean(ExpandingTransformer):
    """Map vectors to items in historical artifact."""

    def __init__(self, vector_file, *args, **kwargs):
        """Initialize vector mapping file."""
        super(HistVectorMean, self).__init__(*args, **kwargs)
        self.vector_file = vector_file
        self.vectors = load_npy(self.vector_file)

    def _reduce_func(self, arr):
        arr = list(itertools.chain.from_iterable(arr))
        n = len(self.vectors)
        return np.mean(
            np.vstack([self.vectors[aa] for aa in arr if aa < n]), axis=0
        ).tolist()


class ListStd(ExpandingTransformer):
    """Expanding mean after merging data based on historical data."""

    def _reduce_func(self, arr):
        arr = list(itertools.chain.from_iterable(arr))
        return np.std(arr)


class SetAggregation(ExpandingTransformer):
    """Count common items between historical and current set."""

    FILL_VALUE = set()

    def _reduce_func(self, arr):
        return set(itertools.chain.from_iterable(arr))


class ListAggregation(ExpandingTransformer):
    """Items in historical data."""

    FILL_VALUE = []

    def __init__(self, n=None, *args, **kwargs):
        """Set n values to extract."""
        super(ListAggregation, self).__init__(*args, **kwargs)
        self.n = n

    def _reduce_func(self, arr):
        if self.n:
            return list(itertools.chain.from_iterable(arr))[-self.n :]
        return list(itertools.chain.from_iterable(arr))


class SetAggregationLast3(ExpandingTransformer):
    """Count common items between historical and current set."""

    FILL_VALUE = set()
    N = 5

    def _reduce_func(self, arr):
        return set(itertools.chain.from_iterable(arr))


class CountCommon(ArrayTransformer):
    """Count common elements between sets."""

    def _transform(self, X):
        return [len(set(row[0]) & set(row[1])) for row in X]


class CountCommonRepeats(ArrayTransformer):
    """Count total counts of all reoccuring items."""

    def _transform(self, X):
        Xt = []
        for row in X:
            hist = row[0]
            curr = set(row[1])
            if len(hist) == 0:
                Xt.append(0)
            else:
                Xt.append(len([h for h in hist if h in curr]))
        return Xt


class Jaccard(ArrayTransformer):
    """Count common elements between sets."""

    def _transform(self, X):
        Xt = []
        eps = 1e-4
        for row in X:
            num = len(set(row[0]) & set(row[1]))
            den = len(set(row[0]) | set(row[1]))
            jac = (eps + num) / (eps + den)
            Xt.append(jac)
        return Xt


class SetLen(ExpandingTransformer):
    """Get no. of items so far."""

    def _reduce_func(self, arr):
        return len(set(itertools.chain.from_iterable(arr)))


class ListLen(ExpandingTransformer):
    """Get no. of items so far."""

    def _reduce_func(self, arr):
        return len(list(itertools.chain.from_iterable(arr)))


class VectorMapper(ArrayTransformer):
    """Map vectors after loading froma a file."""

    def __init__(self, col, vector_file):
        """Initialization."""
        self.col = col
        self.vector_file = vector_file
        self.vectors = load_npy(self.vector_file)

    def _transform(self, X):
        print("Mapping {} vectors".format(self.col))
        return np.vstack([self.vectors[x] for x in X[self.col].values])


class CosineSimilarity(ArrayTransformer):
    """Cosine similarity between two vectors."""

    def _transform(self, X):
        m = X.shape[1] // 2
        return [cosine(row[:m], row[m:]) for row in X]


# class CouponItemMean(ArrayTransformer):
#     def __init__(self, coupon_col, target_col):
#         self.coupon_col = coupon_col
#         self.target_col = target_col
#         self.coupon_item_mean = {}

#     def fit(self, X, y=None):
#         coupon_mean = X.groupby(self.coupon_col)[self.target_col].mean().to_dict()
#         coupon_item = pd.read_csv('data/coupon_item_mapping.csv')
#         coupon_item['coupon_mean'] = coupon_item[self.coupon_col].map(coupon_mean)
#         coupon_item['item_mean'] = coupon_item.item_id.map(coupon_item.groupby('item_id')['coupon_mean'].mean())
#         self.coupon_item_mean = coupon_item.groupby(self.coupon_col)['item_mean'].mean().to_dict()
#         return self

#     def _transform(self, X):
#         return X[self.coupon_col].map(self.coupon_item_mean)
