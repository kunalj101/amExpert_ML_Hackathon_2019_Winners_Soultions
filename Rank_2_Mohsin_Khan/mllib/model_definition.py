from easyml.pipelines import identity
import lightgbm as lgb
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer

from mllib.artifacts import HistoricalArtifact
from mllib.params import FieldNames
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
    AllCountEncoder
)


def get_feature_pipeline(tr_artifact, hist_artifacts, all_data, cachedir='data/'):
    """Define feature transformation pipeline."""
    return make_pipeline(
        make_union(
            identity(
                input_cols=[
                    FieldNames.customer_id,
                    FieldNames.coupon_id,
                    FieldNames.rented,
                    FieldNames.age_range,
                    FieldNames.marital_status,
                    FieldNames.no_of_children,
                    FieldNames.family_size,
                    FieldNames.income_bracket,
                ]
            ),
            make_pipeline(
                SelectCols(cols=[FieldNames.campaign_type]),
                OrdinalEncoder()
            ),
            # make_pipeline(
            #     SelectCols(cols=[FieldNames.cust_cohort]),
            #     OneHotEncoder(handle_unknown='ignore')
            # ),
            make_pipeline(
                GroupCatCatNUnique(FieldNames.campaign_id, FieldNames.customer_id)
            ),
            make_pipeline(
                GroupCatCatNUnique(FieldNames.campaign_id, FieldNames.coupon_id)
            ),
            # make_pipeline(
            #     SelectCols(cols=[FieldNames.campaign_id]),
            #     GroupCatCountEncoder()
            # ),
            make_pipeline(
                ExpandingMean(
                    date_col=FieldNames.campaign_start_date,
                    user_col=FieldNames.customer_id,
                    key_col=FieldNames.target,
                    hist_artifact=tr_artifact,
                ),
            ),
            make_pipeline(
                ExpandingCount(
                    date_col=FieldNames.campaign_start_date,
                    user_col=FieldNames.customer_id,
                    key_col=FieldNames.target,
                    hist_artifact=tr_artifact,
                ),
            ),
            # make_pipeline(
            #     ExpandingMedian(
            #         date_col=FieldNames.campaign_start_date,
            #         user_col=FieldNames.customer_id,
            #         key_col=FieldNames.transaction_day,
            #         hist_artifact=hist_artifacts[0]
            #         )
            # ),
            # make_pipeline(
            #     ExpandingSum(
            #         date_col=FieldNames.campaign_start_date,
            #         user_col=FieldNames.customer_id,
            #         key_col=FieldNames.target,
            #         hist_artifact=tr_artifact,
            #     )
            # ),
            # make_pipeline(
            #     ExpandingCount(
            #         date_col=FieldNames.campaign_start_date,
            #         user_col=FieldNames.customer_id,
            #         key_col=FieldNames.coupon_discount,
            #         hist_artifact=hist_artifacts[0],
            #     )
            # ),
            # make_pipeline(
            #     ExpandingMean(
            #         date_col=FieldNames.campaign_start_date,
            #         user_col=FieldNames.customer_id,
            #         key_col=FieldNames.selling_price,
            #         hist_artifact=hist_artifacts[0],
            #     )
            # ),
            # make_pipeline(
            #     ExpandingMean(
            #         date_col=FieldNames.campaign_start_date,
            #         user_col=FieldNames.customer_id,
            #         key_col=FieldNames.coupon_discount,
            #         hist_artifact=hist_artifacts[1],
            #     )
            # ),
            # make_pipeline(
            #     ExpandingSum(
            #         date_col=FieldNames.campaign_start_date,
            #         user_col=FieldNames.customer_id,
            #         key_col=FieldNames.selling_price,
            #         hist_artifact=hist_artifacts[1],
            #     )
            # ),
            # make_pipeline(
            #     ExpandingMax(
            #         date_col=FieldNames.campaign_start_date,
            #         user_col=FieldNames.customer_id,
            #         key_col=FieldNames.pct_discount,
            #         hist_artifact=hist_artifacts[0],
            #     )
            # ),
            make_pipeline(
                make_union(
                    SetAggregation(
                        date_col=FieldNames.campaign_start_date,
                        user_col=FieldNames.customer_id,
                        key_col=FieldNames.item_set,
                        hist_artifact=hist_artifacts[0],
                    ),
                    SelectCols(cols=[FieldNames.item_set]),
                ),
                CountCommon()
            ),
            make_pipeline(
                make_union(
                    SetAggregation(
                        date_col=FieldNames.campaign_start_date,
                        user_col=FieldNames.customer_id,
                        key_col=FieldNames.item_set,
                        hist_artifact=hist_artifacts[0],
                    ),
                    SelectCols(cols=[FieldNames.item_set]),
                ),
                Jaccard()
            ),
            make_pipeline(
                make_union(
                    SetAggregation(
                        date_col=FieldNames.campaign_start_date,
                        user_col=FieldNames.customer_id,
                        key_col=FieldNames.item_set,
                        hist_artifact=hist_artifacts[1],
                    ),
                    SelectCols(cols=[FieldNames.item_set]),
                ),
                CountCommon(),
            ),
            make_pipeline(
                make_union(
                    SetAggregation(
                        date_col=FieldNames.campaign_start_date,
                        user_col=FieldNames.customer_id,
                        key_col=FieldNames.item_set,
                        hist_artifact=hist_artifacts[1],
                    ),
                    SelectCols(cols=[FieldNames.item_set]),
                ),
                Jaccard(),
                QuantileTransformer(output_distribution='normal')
            ),
            # make_pipeline(
            #     make_union(
            #         SetAggregation(
            #             date_col=FieldNames.campaign_start_date,
            #             user_col=FieldNames.customer_id,
            #             key_col=FieldNames.item_set,
            #             hist_artifact=hist_artifacts[2],
            #         ),
            #         SelectCols(cols=[FieldNames.item_set]),
            #     ),
            #     Jaccard(),
            # ),
            # make_pipeline(
            #     make_union(
            #         SetAggregation(
            #             date_col=FieldNames.campaign_start_date,
            #             user_col=FieldNames.customer_id,
            #             key_col=FieldNames.item_brand,
            #             hist_artifact=hist_artifacts[0],
            #         ),
            #         SelectCols(cols=[FieldNames.item_brand]),
            #     ),
            #     CountCommon(),
            # ),
            make_pipeline(
                make_union(
                    SetAggregation(
                        date_col=FieldNames.campaign_start_date,
                        user_col=FieldNames.customer_id,
                        key_col=FieldNames.item_brand,
                        hist_artifact=hist_artifacts[0],
                    ),
                    SelectCols(cols=[FieldNames.item_brand]),
                ),
                Jaccard(),
            ),
            make_pipeline(
                make_union(
                    SetAggregation(
                        date_col=FieldNames.campaign_start_date,
                        user_col=FieldNames.customer_id,
                        key_col=FieldNames.item_brand,
                        hist_artifact=hist_artifacts[1],
                    ),
                    SelectCols(cols=[FieldNames.item_brand]),
                ),
                Jaccard(),
            ),
            # make_pipeline(
            #     CouponItemMean(coupon_col=FieldNames.coupon_id,
            #                    target_col=FieldNames.target)
            # )
            # make_pipeline(
            #     make_union(
            #         SetAggregation(
            #             date_col=FieldNames.campaign_start_date,
            #             user_col=FieldNames.customer_id,
            #             key_col=FieldNames.item_category,
            #             hist_artifact=hist_artifacts[0],
            #         ),
            #         SelectCols(cols=[FieldNames.item_category]),
            #     ),
            #     Jaccard(),
            # ),
            make_pipeline(
                make_union(
                    SetAggregation(
                        date_col=FieldNames.campaign_start_date,
                        user_col=FieldNames.customer_id,
                        key_col=FieldNames.item_category,
                        hist_artifact=hist_artifacts[1],
                    ),
                    SelectCols(cols=[FieldNames.item_category]),
                ),
                Jaccard(),
            ),
            make_pipeline(
                make_union(
                    SetAggregation(
                        date_col=FieldNames.campaign_start_date,
                        user_col=FieldNames.customer_id,
                        key_col=FieldNames.item_category,
                        hist_artifact=hist_artifacts[2],
                    ),
                    SelectCols(cols=[FieldNames.item_category]),
                ),
                Jaccard(),
            ),
            make_pipeline(
                SetLen(
                        date_col=FieldNames.campaign_start_date,
                        user_col=FieldNames.customer_id,
                        key_col=FieldNames.item_brand,
                        hist_artifact=hist_artifacts[0],
                    ),
            ),
            make_pipeline(
                SelectCols(cols=[FieldNames.campaign_start_date, FieldNames.campaign_end_date]),
                FunctionTransfomer(lambda x: (x.iloc[:, 1] - x.iloc[:, 0]).dt.days)
            ),
            # make_pipeline(
            #     FunctionTransfomer(lambda x: x[FieldNames.item_set].apply(len).values.reshape(-1, 1))
            # ),
            make_pipeline(
                FunctionTransfomer(lambda x: x[FieldNames.item_brand].apply(len).values.reshape(-1, 1))
            ),
            make_pipeline(
                FunctionTransfomer(lambda x: x[FieldNames.item_category].apply(len).values.reshape(-1, 1))
            ),
            make_pipeline(
                ZeroPct(
                    date_col=FieldNames.campaign_start_date,
                    user_col=FieldNames.customer_id,
                    key_col=FieldNames.coupon_discount,
                    hist_artifact=hist_artifacts[0],
                )
            ),
            make_pipeline(
                AllCountEncoder(
                    cols=[FieldNames.customer_id, FieldNames.coupon_id], data=all_data
                )
            ),
            # make_pipeline(
            #     SetMean(
            #         date_col=FieldNames.campaign_start_date,
            #         user_col=FieldNames.customer_id,
            #         key_col=FieldNames.selling_price,
            #         hist_artifact=hist_artifacts[0],
            #     )
            # ),
            # make_pipeline(
            #     ZeroPct(
            #         date_col=FieldNames.campaign_start_date,
            #         user_col=FieldNames.customer_id,
            #         key_col=FieldNames.other_discount,
            #         hist_artifact=hist_artifacts[0],
            #     )
            # )
            # make_pipeline(
            #     VectorMapper(FieldNames.coupon_id, 'data/coupon_vectors_lda.npy')
            # ),
            # make_pipeline(
            #     VectorMapper(FieldNames.coupon_id, 'data/coupon_vectors_svd.npy')
            # ),
            # make_pipeline(
            #     SetLen(
            #             date_col=FieldNames.campaign_start_date,
            #             user_col=FieldNames.customer_id,
            #             key_col=FieldNames.item_set,
            #             hist_artifact=hist_artifacts[0],
            #         ),
            # ),
            # make_pipeline(
            #     SetLen(
            #             date_col=FieldNames.campaign_start_date,
            #             user_col=FieldNames.customer_id,
            #             key_col=FieldNames.item_set,
            #             hist_artifact=hist_artifacts[1],
            #         ),
            # ),
            # make_pipeline(
            #     make_union(
            #         SetAggregationLast3(
            #             date_col=FieldNames.campaign_start_date,
            #             user_col=FieldNames.customer_id,
            #             key_col=FieldNames.item_set,
            #             hist_artifact=hist_artifacts[1],
            #         ),
            #         SelectCols(cols=[FieldNames.item_set]),
            #     ),
            #     Jaccard(),
            # ),
            # make_pipeline(
            #     make_union(
            #         SetAggregation(
            #             date_col=FieldNames.campaign_start_date,
            #             user_col=FieldNames.customer_id,
            #             key_col=FieldNames.item_set,
            #             hist_artifact=hist_artifacts[2],
            #         ),
            #         SelectCols(cols=[FieldNames.item_set]),
            #     ),
            #     Jaccard(),
            # ),
        ),
        make_union(
            FunctionTransfomer(lambda x: x),
            FunctionTransfomer(lambda x: x[:, 13]/(1e-4 + x[:, 15])),
            FunctionTransfomer(lambda x: x[:, 14]/(1e-4 + x[:, 16])),
            FunctionTransfomer(lambda x: x[:, 17]/(1e-4 + x[:, 18])),
            FunctionTransfomer(lambda x: x[:, 19]/(1e-4 + x[:, 20])),
            # FunctionTransfomer(lambda x: x[:, 17]/(1e-4 + x[:, 14])),
            ),
    )


def get_model1():
    """Define model to be fitted on training data."""
    lgb_params = {
        "n_estimators": 5000,
        "boosting_type": "gbdt",
        "num_leaves": 8,
        "max_depth": 4,
        "colsample_bytree": 0.58,
        "metric": None,
        "subsample": 0.8,
        "learning_rate": 0.01,
        "reg_lambda": 0.01,
        "reg_alpha": 0.1,
        "min_data_in_leaf": 150,
        "min_child_samples": 200,
        "max_bin": 255,
        "cat_smooth": 50,
        "max_cat_threshold": 32,
        "cat_l2": 50,
        "seed": 786,
    }
    return lgb.LGBMClassifier(**lgb_params)
