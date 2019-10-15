"""Prepare customer transaction so that it can be used by transformer as historical artifcat."""

import pandas as pd

from mllib.params import FieldNames, FileNames
from mllib.utils import read_csv, save_pickle


def prepare_transactions():
    """Create validation customer transaction data; Aggregate by date and user."""
    cust_transact = read_csv(
        FileNames.transaction, **{"parse_dates": [FieldNames.transaction_date]}
    )
    item_details = read_csv(FileNames.item)
    cust_transact = pd.merge(
        cust_transact, item_details, on=FieldNames.item_id, how="left"
    )
    cust_transact[FieldNames.pct_discount] = (
        cust_transact[FieldNames.coupon_discount]
        / cust_transact[FieldNames.selling_price]
    )
    cust_transact[FieldNames.transaction_dayofweek] = cust_transact[
        FieldNames.transaction_date
    ].dt.dayofweek
    cust_transact_tr = cust_transact.loc[
        cust_transact[FieldNames.transaction_date] <= "2013-05-10"
    ]

    print("Saving to pickle")
    save_pickle(cust_transact, FileNames.transaction_test_v1)
    save_pickle(cust_transact_tr, FileNames.transaction_val_v1)


if __name__ == "__main__":
    prepare_transactions()
