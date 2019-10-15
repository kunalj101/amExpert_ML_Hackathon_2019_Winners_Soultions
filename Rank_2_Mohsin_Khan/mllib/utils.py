"""Public Utilities."""

import joblib

import numpy as np
import pandas as pd
from pathlib import Path

from mllib.params import DATA, FileNames, FieldNames


def read_csv(filename, **kwargs):
    """Read csv in dataframe."""
    return pd.read_csv(str(Path(DATA) / filename), **kwargs)


def read_feather(filename):
    """Read feather file into dataframe."""
    return pd.read_feather(str(Path(DATA) / filename))


def save_pickle(data, filename):
    """Save pickle file."""
    return joblib.dump(data, str(Path(DATA) / filename))


def load_pickle(filename):
    """Load pickle file."""
    return joblib.load(str(Path(DATA) / filename))


def load_npy(filename):
    """Load nupy files."""
    return np.load(str(Path(DATA) / filename))


def save_npy(filename, arr):
    """Save numpy files."""
    return np.save(str(Path(DATA) / filename), arr)


def write_csv(df, filename):
    """Save pandas dataframe as csv."""
    return df.to_csv(str(Path(DATA) / filename), index=False)


def write_feather(df, filename):
    """Dump pandas dataframe to feather file."""
    return df.reset_index(drop=True).to_feather(str(Path(DATA) / filename))


def convert_to_datetime(df, col, format=None, **kwargs):
    """Convert column to pandas datetime format."""
    if not format:
        df[col] = pd.to_datetime(df[col], **kwargs)
    else:
        df[col] = pd.to_datetime(df[col], format=format, **kwargs)
    return df


def read_train_test():
    """Read train and test files."""
    train = read_csv(FileNames.train)
    test = read_csv(FileNames.test)
    return train, test


def split_train_validation(df, val_campaigns=(12, 13)):
    """Split train and validation based on campaign ids."""
    val_rows = df[FieldNames.campaign_id].isin(val_campaigns)
    tr = df.loc[~val_rows]
    val = df.loc[val_rows]
    print('Shape of train and validation data ', tr.shape, val.shape)
    return tr, val
