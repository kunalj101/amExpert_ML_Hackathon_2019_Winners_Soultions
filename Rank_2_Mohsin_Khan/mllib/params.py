"""All constants go here."""

import numpy as np


class FieldNames:
    """Field names in data files."""

    campaign_id = "campaign_id"
    coupon_id = "coupon_id"
    customer_id = "customer_id"
    item_id = "item_id"
    campaign_type = "campaign_type"
    campaign_start_date = "start_date"
    campaign_end_date = "end_date"
    age_range = "age_range"
    marital_status = "marital_status"
    rented = "rented"
    family_size = "family_size"
    no_of_children = "no_of_children"
    income_bracket = "income_bracket"
    item_brand = "brand"
    item_brand_type = "brand_type"
    item_category = "category"
    date_int = "end_date_int"
    transaction_date = "date"
    quantity = "quantity"
    selling_price = "selling_price"
    other_discount = "other_discount"
    coupon_discount = "coupon_discount"
    item_set = "item_set"
    pct_discount = "pct_discount"
    transaction_dayofweek = "dayofweek"
    cust_cohort = "cust_cohort"
    target = "redemption_status"
    idx = "id"


class FileNames:
    """All names of files."""

    demogs = "customer_demographics.csv"
    campaign = "campaign_data.csv"
    coupon_item = "coupon_item_mapping.csv"
    transaction = "customer_transaction_data.csv"
    transaction_test_v0 = "customer_transaction_test_v0.csv"
    transaction_val_v0 = "customer_transaction_val_v0.csv"
    transaction_test_v1 = "customer_transaction_test_v1.csv"
    transaction_val_v1 = "customer_transaction_val_v1.csv"
    item = "item_data.csv"
    train = "train.csv"
    test = "test.csv"
    train_v0 = "train_v0.csv"
    test_v0 = "test_v0.csv"
    tr_v0 = "tr_v0.csv"
    val_v0 = "val_v0.csv"
    tr_v1 = "tr_v1.csv"
    val_v1 = "val_v1.csv"
    train_v2 = "train_v2.csv"
    test_v2 = "test_v2.csv"
    tr_v2 = "tr_v2.csv"
    val_v2 = "val_v2.csv"
    train_artifact = "train_artifact.pkl"
    tr_artifact = "tr_artifact.pkl"
    cust_train_artifact1 = "customer_train_artifact1.pkl"
    cust_train_artifact2 = "customer_train_artifact2.pkl"
    cust_train_artifact3 = "customer_train_artifact3.pkl"
    cust_train_artifact4 = "customer_train_artifact4.pkl"
    cust_tr_artifact1 = "customer_tr_artifact1.pkl"
    cust_tr_artifact2 = "customer_tr_artifact2.pkl"
    cust_tr_artifact3 = "customer_tr_artifact3.pkl"
    cust_tr_artifact4 = "customer_tr_artifact4.pkl"
    train_features_v1 = 'train_features_v1.pkl'
    test_features_v1 = 'test_features_v1.pkl'
    tr_features_v1 = 'tr_features_v1.pkl'
    val_features_v1 = 'val_features_v1.pkl'
    train_features_v2 = 'train_features_v2.pkl'
    test_features_v2 = 'test_features_v2.pkl'
    tr_features_v2 = 'tr_features_v2.pkl'
    val_features_v2 = 'val_features_v2.pkl'
    train_features_v3 = 'train_features_v3.pkl'
    test_features_v3 = 'test_features_v3.pkl'
    tr_features_v3 = 'tr_features_v3.pkl'
    val_features_v3 = 'val_features_v3.pkl'
    coupon_vectors = 'coupon_vectors_nmf.npy'
    item_vectors = 'item_vectors_nmf.npy'
    tr_coupon_nn_data = 'tr_coupon_nn_data.npy'
    tr_customer_hist_nn_data = 'tr_customer_hist_nn_data.npy'
    val_coupon_nn_data = 'val_coupon_nn_data.npy'
    val_customer_hist_nn_data = 'val_customer_hist_nn_data.npy'
    train_coupon_nn_data = 'train_coupon_nn_data.npy'
    train_customer_hist_nn_data = 'train_customer_hist_nn_data.npy'
    test_coupon_nn_data = 'test_coupon_nn_data.npy'
    test_customer_hist_nn_data = 'test_customer_hist_nn_data.npy'


DATA = "data"

AGE_MAP = {
    "18-25": 0,
    "26-35": 1,
    "36-45": 2,
    "46-55": 3,
    "56-70": 4,
    "70+": 5,
    "nan": np.nan,
}

MARITAL_STATUS = {"Single": 0, "Married": 1, "nan": np.nan}

FAMILY_SIZE = {"1": 1, "2": 2, "3": 3, "4": 4, "5+": 5, "nan": np.nan}

NO_OF_CHILDREN = {"0": 0, "1": 1, "2": 2, "3+": 3, "nan": np.nan}

CAMPAIGN_TYPE = ({"X": 0, "Y": 1})
