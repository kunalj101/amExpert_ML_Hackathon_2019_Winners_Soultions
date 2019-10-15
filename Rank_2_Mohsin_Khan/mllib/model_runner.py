"""Run model."""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from mllib.utils import write_csv, read_csv
from mllib.params import DATA, FileNames, FieldNames
from mllib.model_definition import get_feature_pipeline, get_model1
from mllib.artifacts import HistoricalArtifact
from mllib.utils import convert_to_datetime


def get_file_strings(flag="test", version="v0"):
    if flag == "test":
        tr_string = "train"
        te_string = "test"
    else:
        tr_string = "tr"
        te_string = "val"

    tr_fname = "{}_{}".format(tr_string, version)
    te_fname = "{}_{}".format(te_string, version)
    tr_fname = getattr(FileNames, tr_fname)
    te_fname = getattr(FileNames, te_fname)
    return tr_fname, te_fname


def map_coupon_items(df):
    coupon_items = read_csv(FileNames.coupon_item)
    item_data = read_csv(FileNames.item)
    coupon_items = pd.merge(coupon_items, item_data, on=FieldNames.item_id, how="left")
    coupon_items_map = (
        coupon_items.groupby(FieldNames.coupon_id)[FieldNames.item_id]
        .apply(set)
        .to_dict()
    )
    coupon_brand_map = (
        coupon_items.groupby(FieldNames.coupon_id)[FieldNames.item_brand]
        .apply(set)
        .to_dict()
    )
    coupon_category_map = (
        coupon_items.groupby(FieldNames.coupon_id)[FieldNames.item_category]
        .apply(set)
        .to_dict()
    )

    df[FieldNames.item_set] = df[FieldNames.coupon_id].map(coupon_items_map)
    df[FieldNames.item_brand] = df[FieldNames.coupon_id].map(coupon_brand_map)
    df[FieldNames.item_category] = df[FieldNames.coupon_id].map(coupon_category_map)
    return df


# def map_transact_agg(df, flag):
#     transaction_file = 'transaction_{flag}_v0'.format(flag=flag)
#     transaction_file = getattr(FileNames, transaction_file)

#     customer_transaction = read_csv(transaction_file)
#     customer_transaction = group_transactions(customer_transaction)
#     del customer_transaction[FieldNames.item_set]
#     df = pd.merge(df, customer_transaction, on='customer_id', how='left')
#     print(df.shape)
#     return df


def load_train_test(flag="test", version="v0"):
    """Load data."""
    tr_fname, te_fname = get_file_strings(flag, version)
    tr = read_csv(tr_fname)
    te = read_csv(te_fname)

    tr[FieldNames.campaign_start_date] = pd.to_datetime(
        tr[FieldNames.campaign_start_date]
    )
    tr[FieldNames.campaign_end_date] = pd.to_datetime(tr[FieldNames.campaign_end_date])

    te[FieldNames.campaign_start_date] = pd.to_datetime(
        te[FieldNames.campaign_start_date]
    )
    te[FieldNames.campaign_end_date] = pd.to_datetime(te[FieldNames.campaign_end_date])

    tr = map_coupon_items(tr)
    te = map_coupon_items(te)

    # tr = map_transact_agg(tr, flag)
    # te = map_transact_agg(te, flag)

    return tr, te


def group_transactions(transactions):
    grp_cols = [FieldNames.customer_id, FieldNames.transaction_date]
    transactions = (
        transactions.groupby(grp_cols)
        .agg(
            {
                FieldNames.item_id: set,
                FieldNames.selling_price: list,
                FieldNames.coupon_discount: list,
                FieldNames.other_discount: "mean",
                FieldNames.quantity: "sum",
                FieldNames.item_brand: set,
                FieldNames.item_category: set,
                FieldNames.pct_discount: "max",
                FieldNames.transaction_dayofweek: list,
            }
        )
        .reset_index()
    )
    return transactions.rename(columns={FieldNames.item_id: FieldNames.item_set})


def load_artifacts(flag="test", version="v0"):
    """Load artifacts required for transformers."""
    tr_fname, _ = get_file_strings(flag, version)
    tr = read_csv(tr_fname)
    tr = convert_to_datetime(tr, FieldNames.campaign_start_date, **{"dayfirst": True})
    tr_artifact = HistoricalArtifact(
        tr,
        user_field=FieldNames.customer_id,
        date_field=FieldNames.campaign_start_date,
        key_fields=[FieldNames.campaign_id, FieldNames.coupon_id, FieldNames.target],
    )
    del tr
    transaction_file = "transaction_{flag}_v0".format(flag=flag)
    transaction_file = getattr(FileNames, transaction_file)
    transactions = read_csv(transaction_file)
    transactions = convert_to_datetime(transactions, col=FieldNames.transaction_date)
    transactions[FieldNames.transaction_dayofweek] = transactions[
        FieldNames.transaction_date
    ].dt.dayofweek
    transactions[FieldNames.coupon_discount] = np.abs(
        transactions[FieldNames.coupon_discount]
    )
    transactions[FieldNames.other_discount] = np.abs(
        transactions[FieldNames.other_discount]
    )
    transactions[FieldNames.pct_discount] = transactions[FieldNames.coupon_discount] / (
        1 + transactions[FieldNames.selling_price]
    )

    transactions2 = transactions.loc[transactions[FieldNames.coupon_discount] > 0, :]
    transactions2 = group_transactions(transactions2)

    transactions3 = transactions.loc[
        transactions[FieldNames.coupon_discount]
        > transactions[FieldNames.other_discount]
    ]
    transactions3 = group_transactions(transactions3)

    transactions = group_transactions(transactions)
    print(transactions.head(), transactions2.head())
    cust_artifact1 = HistoricalArtifact(
        transactions,
        user_field=FieldNames.customer_id,
        date_field=FieldNames.transaction_date,
        key_fields=[
            FieldNames.item_set,
            FieldNames.item_brand,
            FieldNames.item_category,
            FieldNames.pct_discount,
            FieldNames.selling_price,
            FieldNames.coupon_discount,
            FieldNames.other_discount,
            FieldNames.quantity,
            FieldNames.transaction_dayofweek,
        ],
    )

    cust_artifact2 = HistoricalArtifact(
        transactions2,
        user_field=FieldNames.customer_id,
        date_field=FieldNames.transaction_date,
        key_fields=[
            FieldNames.item_set,
            FieldNames.item_brand,
            FieldNames.item_category,
            FieldNames.pct_discount,
            FieldNames.selling_price,
            FieldNames.coupon_discount,
            FieldNames.other_discount,
            FieldNames.quantity,
        ],
    )

    cust_artifact3 = HistoricalArtifact(
        transactions3,
        user_field=FieldNames.customer_id,
        date_field=FieldNames.transaction_date,
        key_fields=[
            FieldNames.item_set,
            FieldNames.item_brand,
            FieldNames.item_category,
            FieldNames.pct_discount,
            FieldNames.selling_price,
            FieldNames.coupon_discount,
            FieldNames.other_discount,
            FieldNames.quantity,
        ],
    )

    return tr_artifact, cust_artifact1, cust_artifact2, cust_artifact3


def get_model_feature(tr_artifact, cust_artifacts,all_data):
    """Get pipeline and model."""
    feature_pipeline = get_feature_pipeline(tr_artifact, cust_artifacts, all_data)
    model = get_model1()
    return feature_pipeline, model


def get_x_y(feature_pipeline, train, test, flag="test"):
    """Generate x, y using pipeline."""
    y_train = train[FieldNames.target].values

    x_train = feature_pipeline.fit_transform(train, y_train)
    x_test = feature_pipeline.transform(test)

    if flag != "test":
        y_val = test[FieldNames.target].values
        return x_train, x_test, y_train, y_val
    else:
        return x_train, x_test, y_train, None


def fit_model(
    model, x_train, y_train, x_test, y_test, flag="test", categorical_cols=[]
):
    """Fit model on training data."""
    if flag == "test":
        model.fit(x_train, y_train, categorical_feature=categorical_cols)
    else:
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_train, y_train), (x_test, y_test)],
            eval_metric="auc",
            verbose=50,
            early_stopping_rounds=500,
            categorical_feature=categorical_cols,
        )
    return model


def save_numpy(y_preds, model_name, flag):
    """Save numpy predictions as file."""
    filename = "{}_{}.npy".format(model_name, flag)
    np.save(str(Path(DATA) / filename), y_preds)


def save_sub(y_preds, model_name, version="v0"):
    """Save submimssion."""
    filename = "sub_{}.csv".format(model_name)
    _, test = load_train_test("test", version=version)
    sub = test[[FieldNames.idx]]
    sub[FieldNames.target] = y_preds
    write_csv(sub, filename)


def get_scores(y_test, y_preds):
    """Get score using predicted values."""
    score = roc_auc_score(y_test, y_preds)
    print("ROC-AUC : overall - {:8.6f}".format(score))


def save_preds(model, x_test, y_test=None, flag="test", model_name="v0", version="v0"):
    y_preds = model.predict_proba(x_test)[:, 1]
    save_numpy(y_preds, model_name, flag)
    if flag == "test":
        save_sub(y_preds, model_name, version=version)
    else:
        get_scores(y_test, y_preds)


def main():
    """Executor."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--flag", type=str, default="val", choices=["test", "val"])
    parser.add_argument("--n_estimators", type=int, default=10000)
    parser.add_argument("--model_name", type=str, default="v0")
    parser.add_argument("--data_version", type=str, default="v0")
    args = parser.parse_args()
    flag = args.flag

    train, test = load_train_test(flag, version=args.data_version)
    print("loading data succeded")

    tr_artifact, cust_artifact1, cust_artifact2, cust_artifact3 = load_artifacts(
        flag, version=args.data_version
    )
    all_data = pd.concat([train, test])
    feature_pipeline, model = get_model_feature(
        tr_artifact, (cust_artifact1, cust_artifact2, cust_artifact3), all_data
    )
    model.set_params(**{"n_estimators": args.n_estimators})
    print("Loaded feature pipeline and model")

    x_train, x_test, y_train, y_test = get_x_y(feature_pipeline, train, test, flag)
    print("generating features finished")
    print("first five rows of x_train \n", x_train.shape, x_train[:5])
    print("first five rows of x_test \n", x_test.shape, x_test[:5])

    model = fit_model(model, x_train, y_train, x_test, y_test, flag)
    print("Model fitting finished")

    save_preds(
        model,
        x_test,
        y_test,
        flag,
        model_name=args.model_name,
        version=args.data_version,
    )
    print("All Done")


if __name__ == "__main__":
    main()
