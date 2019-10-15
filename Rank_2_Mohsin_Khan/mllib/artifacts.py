"""Define artifacts to be used by transformers."""

from collections import defaultdict
import time

import numpy as np
from pathlib import Path
import pandas as pd
from mllib.params import DATA


class HistoricalArtifact:
    """Efficient data structure for aggregate data before any date in history."""

    def __init__(self, df, user_field, date_field, key_fields):
        """Initialization."""
        # self.db_file = str((Path(DATA)) / db_file)
        self.df = df
        self.user_field = user_field
        self.date_field = date_field
        self.key_fields = key_fields
        # self.parse_date = parse_date
        # self.parser_kwargs = parser_kwargs
        # if not self.parser_kwargs:
        #    self.parser_kwargs = {}

        self.user_time_arr = defaultdict(list)
        self.user_key_dict = defaultdict(list)

        self._prepare_db()

    def user_date_value(self, user, date, key_field, reduce_func, n=None):
        """For all users apply func to last n values of key before given date.

        Args:
            func: Input to functions will be a array of values sorted in time.
              Funtion should reduce the array to sinple value. Most of the time function could be cached.
              Even numba functions are accepted.

        """
        if key_field not in set(self.key_fields):
            raise ValueError("Key not from intialized key fields")
        date = self._date_to_int(date)
        prev_date_index = self._get_date_index(user, date)
        if prev_date_index <= 0:
            return None
        start_idx = 0
        if n:
            start_idx = min(0, prev_date_index - n)
        key_values = self._get_key_values(user, key_field)[start_idx:prev_date_index]
        if key_values:
            return reduce_func(key_values)
        return None

    def _date_to_int(self, date):
        if not isinstance(date, int):
            return int(date)
        return date

    def _prepare_db(self):
        start_time = time.time()
        # cols_to_use = [self.user_field] + [self.date_field] + self.key_fields
        # datetime_cols = None

        # if self.parse_date:
        #     datetime_cols = [self.date_field]

        # df = pd.read_csv(
        #     self.db_file,
        #     parse_dates=datetime_cols,
        #     usecols=cols_to_use,
        #     **self.parser_kwargs
        # )

        cols_to_idx = {col: i for i, col in enumerate(self.df.columns)}
        df = self.df.sort_values(by=[self.user_field, self.date_field])
        records = df.to_records(index=False)
        for row in records:
            date_value = row[cols_to_idx[self.date_field]]
            date_value = self._date_to_int(date_value)

            user_value = row[cols_to_idx[self.user_field]]
            self.user_time_arr[user_value].append(date_value)
            for key in self.key_fields:
                key_value = row[cols_to_idx[key]]
                self.user_key_dict[(user_value, key)].append(key_value)

        print("Preparing db took {:9.6f}".format(time.time() - start_time))

    def _get_date_index(self, user, date):
        user_dates = self.user_time_arr.get(user, None)
        if user_dates:
            return np.searchsorted(user_dates, date)
        return -1

    def _get_key_values(self, user, key_field):
        return self.user_key_dict.get((user, key_field), None)
