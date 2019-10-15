"""Prepare data."""

import joblib

import pandas as pd

from mllib.utils import read_train_test, read_csv, save_pickle
from mllib.params import (
    FileNames,
    FieldNames,
    AGE_MAP,
    MARITAL_STATUS,
    FAMILY_SIZE,
    NO_OF_CHILDREN,
    CAMPAIGN_TYPE,
)


def map_to_float(df, col, mapping):
    return df[col].astype(str).map(mapping).astype(float)


def main():
    """Load train and test, map additional data, split validation and save as pickle."""
    print("Read train and test files")
    train, test = read_train_test()

    print("Read and map campaign start and end dates")
    kws = {
        "parse_dates": [FieldNames.campaign_start_date, FieldNames.campaign_end_date],
        "dayfirst": True,
    }
    campaign_data = read_csv(FileNames.campaign, **kws)
    train = pd.merge(train, campaign_data, on="campaign_id", how="left")
    test = pd.merge(test, campaign_data, on="campaign_id", how="left")

    print("Read and map demograhics data")
    demog_data = read_csv(FileNames.demogs)
    train = pd.merge(train, demog_data, on="customer_id", how="left")
    test = pd.merge(test, demog_data, on="customer_id", how="left")
    for col, mapping in [
        (FieldNames.age_range, AGE_MAP),
        (FieldNames.marital_status, MARITAL_STATUS),
        (FieldNames.family_size, FAMILY_SIZE),
        (FieldNames.no_of_children, NO_OF_CHILDREN),
        (FieldNames.campaign_type, CAMPAIGN_TYPE),
    ]:
        train[col] = map_to_float(train, col, mapping)
        test[col] = map_to_float(test, col, mapping)

    print("Read coupon and item details and merge them")
    coupon_data = read_csv(FileNames.coupon_item)
    item_data = read_csv(FileNames.item)
    coupon_data = pd.merge(coupon_data, item_data, on="item_id", how="left")

    print("Map coupon details to train")
    coupon_grouped = coupon_data.groupby("coupon_id").agg(
        {"item_id": list, "brand": list, "brand_type": list, "category": list}
    )
    train = pd.merge(train, coupon_grouped, on="coupon_id", how="left")
    test = pd.merge(test, coupon_grouped, on="coupon_id", how="left")

    train = train.rename(columns={'item_id': FieldNames.item_set})
    test = test.rename(columns={'item_id': FieldNames.item_set})

    print("split train --> tr and val")
    tr = train.loc[~train[FieldNames.campaign_id].isin([11, 12, 13])]
    val = train.loc[train[FieldNames.campaign_id].isin([11, 12, 13])]

    print("save as pickle")
    save_pickle(train, FileNames.train_v2)
    save_pickle(test, FileNames.test_v2)
    save_pickle(tr, FileNames.tr_v2)
    save_pickle(val, FileNames.val_v2)


if __name__ == "__main__":
    main()
