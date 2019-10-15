import itertools

import pandas as pd
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer

from mllib.artifacts import HistoricalArtifact
from mllib.params import FieldNames, FileNames
from mllib.transformers import (
    SelectCols,
    FunctionTransfomer,
    CatCatUniqueCount,
    GroupCatCatNUnique,
    GroupCatCountEncoder,
    ExpandingCount,
    ExpandingMean,
    ExpandingSum,
    SetAggregation,
    SetAggregationLast3,
    CountCommon,
    Jaccard,
    SetLen,
    VectorMapper,
    ExpandingMax,
    ExpandingMedian,
    ZeroPct,
    SetMean,
    ListStd,
    CountCommonRepeats,
    AllCountEncoder,
)
from mllib.utils import load_pickle, save_pickle


def get_feature_pipeline(tr_artifact, hist_artifacts, all_data):
    """Feature generation pipeline."""
    hist_n = 3  # len(hist_artifacts)
    tr_artifact_kws = {
        "date_col": FieldNames.campaign_start_date,
        "user_col": FieldNames.customer_id,
        "key_col": FieldNames.target,
        "hist_artifact": tr_artifact,
    }
    hist_cols = [FieldNames.item_set, FieldNames.item_brand, FieldNames.item_category]
    hist_cols2 = [
        FieldNames.coupon_discount,
        FieldNames.other_discount,
        FieldNames.pct_discount,
        FieldNames.quantity,
        FieldNames.selling_price,
    ]
    return make_pipeline(
        make_union(
            # Numerical features directly available
            make_pipeline(
                SelectCols(
                    cols=[
                        FieldNames.customer_id,
                        FieldNames.coupon_id,
                        FieldNames.age_range,
                        FieldNames.marital_status,
                        FieldNames.family_size,
                        FieldNames.no_of_children,
                        FieldNames.income_bracket,
                        FieldNames.campaign_type,
                    ]
                ),
                FunctionTransfomer(lambda x: x),
            ),
            # coupon-no. of unique item attributes
            make_union(
                *[
                    make_pipeline(
                        SelectCols(cols=[col]),
                        FunctionTransfomer(
                            lambda X: [len(set(x)) for x in X.values.flatten().tolist()]
                        ),
                    )
                    for col in [
                        FieldNames.item_set,
                        FieldNames.item_brand,
                        FieldNames.item_brand_type,
                        FieldNames.item_category,
                    ]
                ],
                verbose=True
            ),
            # Campaign id features
            make_union(
                *[
                    GroupCatCatNUnique(FieldNames.campaign_id, col2)
                    for col2 in [FieldNames.customer_id, FieldNames.coupon_id]
                ],
                verbose=True
            ),
            # Customer id expanding mean, count, sum
            make_pipeline(ExpandingMean(**tr_artifact_kws)),
            make_pipeline(ExpandingCount(**tr_artifact_kws)),
            make_pipeline(ExpandingSum(**tr_artifact_kws)),
            # Count items common between current coupon and historical customer transactions
            make_union(
                *[
                    make_pipeline(
                        make_union(
                            SetAggregation(
                                date_col=FieldNames.campaign_start_date,
                                user_col=FieldNames.customer_id,
                                key_col=col,
                                hist_artifact=hist_artifacts[i],
                            ),
                            SelectCols(cols=[col]),
                        ),
                        CountCommon(),
                    )
                    for col, i in itertools.product(hist_cols, range(hist_n))
                ]
            ),
            make_union(
                *[
                    make_pipeline(
                        make_union(
                            SetAggregation(
                                date_col=FieldNames.campaign_start_date,
                                user_col=FieldNames.customer_id,
                                key_col=col,
                                hist_artifact=hist_artifacts[i],
                            ),
                            SelectCols(cols=[col]),
                        ),
                        Jaccard(),
                    )
                    for col, i in itertools.product(hist_cols, range(hist_n))
                ],
                verbose=True
            ),
            make_union(
                *[
                    make_pipeline(
                        make_union(
                            SetAggregation(
                                date_col=FieldNames.campaign_start_date,
                                user_col=FieldNames.customer_id,
                                key_col=col,
                                hist_artifact=hist_artifacts[i],
                            ),
                            SelectCols(cols=[col]),
                        ),
                        CountCommonRepeats(),
                    )
                    for col, i in itertools.product(hist_cols, range(hist_n))
                ]
            ),
            # campaign length
            make_pipeline(
                SelectCols(
                    cols=[FieldNames.campaign_start_date, FieldNames.campaign_end_date]
                ),
                FunctionTransfomer(lambda x: (x.iloc[:, 1] - x.iloc[:, 0]).dt.days),
            ),
            # coupon discount, other dicount, selling price and quantity aggregations
            make_union(
                *[
                    ExpandingMean(
                        date_col=FieldNames.campaign_end_date,
                        user_col=FieldNames.customer_id,
                        key_col=col,
                        hist_artifact=hist_artifacts[i],
                    )
                    for col, i in itertools.product(hist_cols2, range(hist_n))
                ]
            ),
            make_pipeline(
                GroupCatCountEncoder(
                    cols=[FieldNames.customer_id, FieldNames.campaign_id]
                )
            ),
            make_pipeline(
                AllCountEncoder(
                    cols=[FieldNames.customer_id, FieldNames.coupon_id], data=all_data
                )
            ),
        ),
        make_union(
            FunctionTransfomer(lambda x: x),
            # Ratios
            make_pipeline(
                make_union(
                    *[
                        FunctionTransfomer(lambda x: x[:, i] / x[:, j])
                        for (i, j) in itertools.product(range(16, 34), range(16, 34))
                    ],
                    verbose=True
                )
            ),
        ),
    )


def get_feature_names(hist_n):
    cols1 = [
        FieldNames.customer_id,
        FieldNames.coupon_id,
        FieldNames.age_range,
        FieldNames.marital_status,
        FieldNames.family_size,
        FieldNames.no_of_children,
        FieldNames.income_bracket,
        FieldNames.campaign_type,
    ]
    cols2 = [
        "coupon_nuniq_{}".format(col)
        for col in [
            FieldNames.item_id,
            FieldNames.item_brand,
            FieldNames.item_brand_type,
            FieldNames.item_category,
        ]
    ]
    cols3 = [
        "campaign_nuniq_{}".format(col)
        for col in [FieldNames.customer_id, FieldNames.coupon_id]
    ]
    cols4 = ["customer_exp_{}".format(stat) for stat in ["mean", "count", "sum"]]
    hist_cols = [FieldNames.item_set, FieldNames.item_brand, FieldNames.item_category]
    hist_cols2 = [
        FieldNames.coupon_discount,
        FieldNames.other_discount,
        FieldNames.pct_discount,
        FieldNames.quantity,
        FieldNames.selling_price,
    ]

    cols5 = [
        "common_{}_{}".format(col, i)
        for col, i in itertools.product(hist_cols, range(hist_n))
    ]
    cols6 = [
        "jaccard_{}_{}".format(col, i)
        for col, i in itertools.product(hist_cols, range(hist_n))
    ]
    cols7 = [
        "common_repeats_{}_{}".format(col, i)
        for col, i in itertools.product(hist_cols, range(hist_n))
    ]
    cols8 = ["campaign_length"]
    cols9 = [
        "coupon_details_{}_{}".format(col, i)
        for col, i in itertools.product(hist_cols2, range(hist_n))
    ]
    cols10 = ['customer_campaign_count', 'customer_coupon_count_all']
    feats_base = cols1 + cols2 + cols3 + cols4 + cols5 + cols6 + cols7 + cols8 + cols9 + cols10
    cols11 = [
        "{}_by_{}".format(feats_base[i], feats_base[j])
        for i, j in itertools.product(range(16, 34), range(16, 34))
    ]
    return feats_base + cols11


def generate_features(flag):
    if flag == "test":
        tr_artifact_file = FileNames.train_artifact
        hist_artifact_files = [
            FileNames.cust_train_artifact1,
            FileNames.cust_train_artifact2,
            FileNames.cust_train_artifact3,
            FileNames.cust_train_artifact4,
        ]
        tr_file = FileNames.train_v2
        te_file = FileNames.test_v2
        tr_save_file = FileNames.train_features_v1
        te_save_file = FileNames.test_features_v1
    elif flag == "val":
        tr_artifact_file = FileNames.tr_artifact
        hist_artifact_files = [
            FileNames.cust_tr_artifact1,
            FileNames.cust_tr_artifact2,
            FileNames.cust_tr_artifact3,
            FileNames.cust_tr_artifact4,
        ]
        tr_file = FileNames.tr_v2
        te_file = FileNames.val_v2
        tr_save_file = FileNames.tr_features_v1
        te_save_file = FileNames.val_features_v1
    else:
        print("flag not VALD!")
    tr_artifact = load_pickle(tr_artifact_file)
    hist_artifacts = [load_pickle(hist_file) for hist_file in hist_artifact_files]
    columns = get_feature_names(3)
    tr_data = load_pickle(tr_file)
    te_data = load_pickle(te_file)
    all_data = pd.concat([tr_data, te_data])
    pipeline = get_feature_pipeline(tr_artifact, hist_artifacts, all_data)

    x_tr = pipeline.fit_transform(tr_data)
    x_te = pipeline.transform(te_data)

    x_tr = pd.DataFrame(x_tr, columns=columns)
    x_te = pd.DataFrame(x_te, columns=columns)
    x_tr[FieldNames.target] = tr_data[FieldNames.target].values
    if flag == "val":
        x_te[FieldNames.target] = te_data[FieldNames.target].values
    save_pickle(x_tr, tr_save_file)
    save_pickle(x_te, te_save_file)


def main():
    generate_features("test")
    generate_features("val")


if __name__ == "__main__":
    main()
