import pandas as pd
import numpy as np

from mllib.params import FileNames, FieldNames
from mllib.utils import convert_to_datetime
from mllib.utils import read_train_test, read_csv, split_train_validation
from mllib.utils import write_csv

AGE_MAP = {'18-25': 0,
           '26-35': 1,
           '36-45': 2,
           '46-55': 3,
           '56-70': 4,
           '70+': 5,
           'nan': np.nan}

MARITAL_STATUS = {'Single': 0,
                  'Married': 1,
                  'nan': np.nan}

FAMILY_SIZE = {'1': 1,
               '2': 2,
               '3': 3,
               '4': 4,
               '5+': 5,
               'nan': np.nan}

NO_OF_CHILDREN = {'0': 0,
                  '1': 1,
                  '2': 2,
                  '3+': 3,
                  'nan': np.nan}


def map_to_float(df, col, mapping):
    return df[col].astype(str).map(mapping).astype(float)


def main():
    train, test = read_train_test()
    demog_df = read_csv(FileNames.demogs)
    demog_df[FieldNames.no_of_children] = demog_df[FieldNames.no_of_children].fillna(0)

    camp_data = read_csv(FileNames.campaign)
    camp_data = convert_to_datetime(camp_data, FieldNames.campaign_start_date, **{'dayfirst': True})
    camp_data = convert_to_datetime(camp_data, FieldNames.campaign_end_date, **{'dayfirst': True})
    camp_data[FieldNames.date_int] = (camp_data[FieldNames.campaign_end_date].astype(int)/10**12).astype(int)

    print([demog_df[col].unique() for col in demog_df.columns])
    train = pd.merge(train, demog_df, on='customer_id', how='left')
    test = pd.merge(test, demog_df, on='customer_id', how='left')

    train = pd.merge(train, camp_data, on='campaign_id', how='left')
    test = pd.merge(test, camp_data, on='campaign_id', how='left')

    for col, mapping in [(FieldNames.age_range, AGE_MAP),
                         (FieldNames.marital_status, MARITAL_STATUS),
                         (FieldNames.family_size, FAMILY_SIZE),
                         (FieldNames.no_of_children, NO_OF_CHILDREN)]:
        train[col] = map_to_float(train, col, mapping)
        test[col] = map_to_float(test, col, mapping)

    tr, val = split_train_validation(train, val_campaigns=(12, 13))
    write_csv(train, FileNames.train_v0)
    write_csv(test, FileNames.test_v0)
    write_csv(tr, FileNames.tr_v0)
    write_csv(val, FileNames.val_v0)

    tr, val = split_train_validation(train, val_campaigns=(26, 27, 28, 29, 30))
    write_csv(tr, FileNames.tr_v1)
    write_csv(val, FileNames.val_v1)

    cust_transactions = read_csv(FileNames.transaction)
    item_data = read_csv(FileNames.item)
    cust_transactions = pd.merge(cust_transactions, item_data, on=FieldNames.item_id, how='left')
    cust_transactions = convert_to_datetime(cust_transactions, FieldNames.transaction_date)
    cust_transactions_v0 = cust_transactions.loc[cust_transactions[FieldNames.transaction_date] <= '2013-05-10']
    write_csv(cust_transactions, FileNames.transaction_test_v0)
    write_csv(cust_transactions_v0, FileNames.transaction_val_v0)


if __name__ == '__main__':
    main()
