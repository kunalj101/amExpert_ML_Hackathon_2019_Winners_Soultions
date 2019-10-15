"""Prepare artifacts required for feature generation."""

import numpy as np

from mllib.artifacts import HistoricalArtifact
from mllib.params import FieldNames, FileNames
from mllib.utils import save_pickle, load_pickle


def group_transactions(transactions):
    """Group transactions by date and customer and aggregate all columns as list."""
    grp_cols = [FieldNames.customer_id, FieldNames.transaction_date]
    transactions = transactions.groupby(grp_cols).agg(list).reset_index()
    print(transactions.head())
    return transactions.rename(columns={FieldNames.item_id: FieldNames.item_set})


def save_train_artifact(flag):
    """Create artifact using training data."""
    if flag == 'test':
        inp_file = FileNames.train_v2
        save_file = FileNames.train_artifact
    elif flag == 'val':
        inp_file = FileNames.tr_v2
        save_file = FileNames.tr_artifact

    tr = load_pickle(inp_file)
    tr_artifact = HistoricalArtifact(
        tr,
        user_field=FieldNames.customer_id,
        date_field=FieldNames.campaign_start_date,
        key_fields=[
            FieldNames.campaign_id,
            FieldNames.coupon_id,
            FieldNames.target,
            FieldNames.item_category,
        ],
    )
    save_pickle(tr_artifact, save_file)


def _get_transaction_artifact(transactions):
    cust_artifact = HistoricalArtifact(
        transactions,
        user_field=FieldNames.customer_id,
        date_field=FieldNames.transaction_date,
        key_fields=[
            FieldNames.item_set,
            FieldNames.item_brand,
            FieldNames.item_brand_type,
            FieldNames.item_category,
            FieldNames.pct_discount,
            FieldNames.selling_price,
            FieldNames.coupon_discount,
            FieldNames.other_discount,
            FieldNames.quantity,
            FieldNames.transaction_dayofweek,
        ],
    )
    return cust_artifact


def save_transaction_artifact(flag):
    """Sace artifacts for customer transactions with different conditions."""
    if flag == 'test':
        inp_file = FileNames.transaction_test_v1
        save_file1 = FileNames.cust_train_artifact1
        save_file2 = FileNames.cust_train_artifact2
        save_file3 = FileNames.cust_train_artifact3
        save_file4 = FileNames.cust_train_artifact4
    elif flag == 'val':
        inp_file = FileNames.transaction_val_v1
        save_file1 = FileNames.cust_tr_artifact1
        save_file2 = FileNames.cust_tr_artifact2
        save_file3 = FileNames.cust_tr_artifact3
        save_file4 = FileNames.cust_tr_artifact4
    else:
        print('flag not VALID!')

    transactions = load_pickle(inp_file)
    transactions_grp = group_transactions(transactions)
    artifact = _get_transaction_artifact(transactions_grp)
    save_pickle(artifact, save_file1)
    del artifact, transactions_grp
    print("Customer artifact 1 done!")

    transactions2 = transactions.loc[
        np.abs(transactions[FieldNames.coupon_discount]) > 0
    ]
    transactions_grp2 = group_transactions(transactions2)
    artifact = _get_transaction_artifact(transactions_grp2)
    save_pickle(artifact, save_file2)
    del transactions2, transactions_grp2, artifact
    print("Customer artifact 2 done!")

    transactions3 = transactions.loc[
        (np.abs(transactions[FieldNames.coupon_discount]) > 0)
        & (np.abs(transactions[FieldNames.other_discount]) > 0)
    ]
    transactions_grp3 = group_transactions(transactions3)
    artifact = _get_transaction_artifact(transactions_grp3)
    save_pickle(artifact, save_file3)
    del transactions3, transactions_grp3, artifact
    print("Customer artifact 3 done!")

    transactions4 = transactions.loc[
        (
            np.abs(transactions[FieldNames.coupon_discount])
            > np.abs(transactions[FieldNames.other_discount])
        )
    ]
    transactions_grp4 = group_transactions(transactions4)
    artifact = _get_transaction_artifact(transactions_grp4)
    save_pickle(artifact, save_file4)
    del transactions4, artifact
    print("Customer artifact 4 done!")


def main():
    """Save all."""
    save_train_artifact('test')
    save_train_artifact('val')
    save_transaction_artifact('test')
    save_transaction_artifact('val')


if __name__ == '__main__':
    main()
