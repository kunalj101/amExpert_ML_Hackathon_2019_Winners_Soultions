import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.exceptions import NotFittedError
import numpy as np
from scipy import stats
# from astroML.density_estimation import bayesian_blocks
import pickle
from custom_fold_generator import CustomFolds

def get_categorical_column_indexes(df, subset_cols = None, ignore_cols = None, threshold = 10):
    '''
    Function that returns categorical columns indexes from the dataFrame
    Input:Target

        df: pandas dataFrame
        subset_cols: list of columns to filter categorical columns from
        ignore: list of columns to ignore from categorical columns
    returns:
        list with indexes of the Categorical columns from the list of dataframe columns
    '''
    def column_index(df, query_cols):
        cols = df.columns.values
        sidx = np.argsort(cols)
        return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
    # print df.shape
    # print df.head()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()+[x for x in df.select_dtypes(include=[np.number]).columns if df[x].nunique() < threshold]
    if subset_cols is not None and type(subset_cols) == list:
        cat_cols = [x for x in cat_cols if x in subset_cols]
    if ignore_cols is not None and type(ignore_cols) == list:
        cat_cols = [x for x in cat_cols if x not in ignore_cols]
    return column_index(df, cat_cols)

def get_numerical_column_indexes(df, subset_cols = None, ignore_cols = None, threshold = 10):
    '''
    Function that returns numerical columns indexes from the dataFrame
    Input:
        df: pandas dataFrame
        subset_cols: list of columns to filter categorical columns from
        ignore: list of columns to ignore from categorical columns
    returns:
        list with indexes of the Categorical columns from the list of dataframe columns
    '''
    def column_index(df, query_cols):
        cols = df.columns.values
        sidx = np.argsort(cols)
        return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
    cat_cols = [x for x in df.select_dtypes(include=[np.number]).columns if df[x].nunique() >= threshold]
    if subset_cols is not None and type(subset_cols) == list:
        cat_cols = [x for x in cat_cols if x in subset_cols]
    if ignore_cols is not None and type(ignore_cols) == list:
        cat_cols = [x for x in cat_cols if x not in ignore_cols]
    return column_index(df, cat_cols)


class Encoding(BaseEstimator):
    categorical_columns = None
    return_df = False
    random_state = None
    threshold = 50

    def __init__(self):
        pass

    def convert_input(self, X):
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, list):
                X = pd.DataFrame(np.array(X))
            elif isinstance(X, (np.generic, np.ndarray, pd.Series)):
                X = pd.DataFrame(X)
            else:
                raise ValueError('Unexpected input type: %s' % (str(type(X))))
            X = X.apply(lambda x: pd.to_numeric(x, errors='ignore'))
        x = X.copy(deep = True)
        return x

    def get_categorical_columns(self, X, categorical_columns=None):
        # if categorical_columns is None then Auto interpret categorical columns,
        # else if categorical_columns is string 'all' then treat all columns as categorical columns
        # else return categorical_columns if categorical_columns is a list
        if categorical_columns is None:
            return X.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_columns == 'all':
            return X.columns.tolist()
        return categorical_columns

    def get_numerical_columns(self,X):
        temp_x=X[X.columns[X.nunique()<=self.threshold]]
        col_names=temp_x.columns[temp_x.dtypes!='object']
        return col_names

    def apply_encoding(self, X_in, encoding_dict):
        X = self.convert_input(X_in)
        for col in self.categorical_columns:
            if col in encoding_dict:
                freq_dict = encoding_dict[col]
                X[col] = X[col].apply(lambda x: freq_dict[x] if x  in freq_dict else np.nan)
        return X

    def create_encoding_dict(self, X, y):
        return {}

    def fit(self, X, y=None):
        if X is None:
            raise ValueError("Input array is required to call fit method!")
        X = self.convert_input(X)
        self.encoding_dict = self.create_encoding_dict(X, y)
        return self

    def transform(self, X, return_df=True):
        df = self.apply_encoding(X, self.encoding_dict)
        if self.return_df or return_df:
            return df
        else:
            return df.values

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = self.convert_input(X)
        for col in self.categorical_columns:
            freq_dict = self.encoding_dict[col]
            for key, val in list(freq_dict.items()):
                X.loc[X[col] == val, col] = key
        if self.return_df:
            return X
        else:
            return X.values

    def save_encoding(self, file_name):
        assert self.encoding_dict and len(self.encoding_dict.keys()) > 0, "Cannot save a model that is not fitted"
        assert file_name, "file_name cannot be None"
        with open(file_name, "wb") as out_file:
            pickle.dump(self.encoding_dict, out_file)
            return file_name

    def load_encoding(self, file_name):
        assert file_name, "file_name cannot be None"
        if file_name:
            self.encoding_dict = pickle.load(open(file_name, "rb"))
            return True
        return False

class LabelEncoding(Encoding):
    '''
    class to perform FreqeuncyEncoding on Categorical Variables
    Initialization Variabes:
    categorical_columns: list of categorical columns from the dataframe
    or list of indexes of caategorical columns for numpy ndarray
    return_df: boolean
        if True: returns pandas dataframe on transformation
        else: return numpy ndarray
    '''
    def __init__(self, categorical_columns = None, return_df = False, **kwargs):
        self.categorical_columns = categorical_columns
        self.return_df = return_df

    def create_encoding_dict(self, X, y):
        encoding_dict = {}
        self.categorical_columns = self.get_categorical_columns(X, self.categorical_columns)
        #print self.categorical_columns
        for col in self.categorical_columns:
            encoding_dict.update({col: pd.unique(X[col]).tolist()})
        return encoding_dict

    def apply_encoding(self, X_in, encoding_dict):
        X = self.convert_input(X_in)
        for col in self.categorical_columns:
            values = encoding_dict[col]
            X[col] = X[col].apply(lambda x: values.index(x) + 1 if x in values else 0)
        return X


class FreqeuncyEncoding(Encoding):
    '''
    class to perform FreqeuncyEncoding on Categorical Variables
    Initialization Variabes:
    categorical_columns: list of categorical columns from the dataframe
    or list of indexes of caategorical columns for numpy ndarray
    return_df: boolean
        if True: returns pandas dataframe on transformation
        else: return numpy ndarray
    '''
    def __init__(self, normalize=1, categorical_columns = None, return_df = False, **kwargs):
        self.categorical_columns = categorical_columns
        self.return_df = return_df
        self.normalize = normalize

    def get_params(self):
        return {
            "categorical_columns": self.categorical_columns,
            "normalize": self.normalize,
            "return_df": self.return_df
        }

    def create_encoding_dict(self, X, y):
        encoding_dict = {}
        self.categorical_columns = self.get_categorical_columns(X, self.categorical_columns)
        for col in self.categorical_columns:
            encoding_dict.update({col: X[col].value_counts(self.normalize).to_dict()})
        return encoding_dict

    def apply_encoding(self, X_in, encoding_dict):
        X = self.convert_input(X_in)
        for col in self.categorical_columns:
            if col in encoding_dict:
                X[col] = X[col].apply(lambda x: encoding_dict[col].get(x, 0))
        return X

