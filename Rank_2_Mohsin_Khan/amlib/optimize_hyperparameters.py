import logging

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import lightgbm as lgb
import numpy as np
import skopt
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
from scipy.optimize import OptimizeResult
from sklearn.preprocessing import QuantileTransformer

from mllib.params import FileNames
from mllib.utils import load_pickle

logging.basicConfig(level=logging.INFO)


STATIC_PARAMS = {
    "boosting": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "max_depth": 2,
    "verbosity": -1,
}
HOST = "127.0.0.1"
PORT = 8999
RUN_ID = "4"
WORKING_DIRECTORY = "."

HPO_PARAMS = {
    "n_calls": 100,
    "eta": 3,
    "min_budget": 1,
    "max_budget": 60,
    "num_samples": 64,
    "top_n_percent": 10,
    "min_bandwidth": 1e-3,
    "bandwidth_factor": 3,
}


def make_x_y(x_tr, x_val, flag="val"):
    y_tr = x_tr["redemption_status"].values
    del x_tr["redemption_status"]

    if flag == "val":
        y_val = x_val["redemption_status"].values
        del x_val["redemption_status"]
    else:
        y_val = None
    return x_tr, y_tr, x_val, y_val


def get_rank_features(df):
    df["cust_coupon_rank1"] = (
        df.groupby(["customer_id", "campaign_id"])["common_item_set_0"].rank("max")
        / df["customer_campaign_count"]
    )
    df["cust_coupon_rank2"] = (
        df.groupby(["customer_id", "campaign_id"])["common_item_set_1"].rank("max")
        / df["customer_campaign_count"]
    )
    df["cust_coupon_rank3"] = (
        df.groupby(["customer_id", "campaign_id"])["common_item_set_2"].rank("max")
        / df["customer_campaign_count"]
    )
    df["customer_rank1"] = (
        df.groupby(["customer_id"])["common_item_set_0"].rank("max")
        / df.groupby("customer_id").size()
    )
    df["customer_rank2"] = (
        df.groupby(["customer_id"])["common_brand_1"].rank("max")
        / df.groupby("customer_id").size()
    )
    df["customer_rank3"] = (
        df.groupby(["customer_id"])["common_category_1"].rank("max")
        / df.groupby("customer_id").size()
    )

    # df['customer_rank2'] = df.groupby(['customer_id'])['common_item_set_1'].rank('max')/df.groupby('customer_id').size()
    # df['customer_rank3'] = df.groupby(['customer_id'])['common_item_set_2'].rank('max')/df.groupby('customer_id').size()
    df["campaign_rank1"] = (
        df.groupby(["campaign_id"])["common_item_set_0"].rank("max")
        / df.groupby("campaign_id").size()
    )
    df["campaign_rank2"] = (
        df.groupby(["campaign_id"])["common_brand_0"].rank("max")
        / df.groupby("campaign_id").size()
    )

    # df['campaign_rank2'] = df.groupby(['campaign_id'])['common_item_set_1'].rank('max')/df.groupby('campaign_id').size()
    # df['campaign_rank3'] = df.groupby(['campaign_id'])['common_item_set_2'].rank('max')/df.groupby('campaign_id').size()
    df["coupon_rank1"] = (
        df.groupby(["coupon_id"])["common_item_set_0"].rank("max")
        / df.groupby("coupon_id").size()
    )
    # df['coupon_rank2'] = df.groupby(['coupon_id'])['common_item_set_1'].rank('max')/df.groupby('coupon_id').size()

    return df


def load_data(flag="val"):
    if flag == "val":
        x_tr = load_pickle(FileNames.tr_features_v1)
        x_val = load_pickle(FileNames.val_features_v1)
    elif flag == "test":
        x_tr = load_pickle(FileNames.train_features_v1)
        x_val = load_pickle(FileNames.test_features_v1)

    return make_x_y(x_tr, x_val, flag=flag)


def map_campign_id(x_tr, x_val, flag="val"):
    if flag == "val":
        tr = load_pickle(FileNames.tr_v2)
        val = load_pickle(FileNames.val_v2)
    elif flag == "test":
        tr = load_pickle(FileNames.train_v2)
        val = load_pickle(FileNames.test_v2)

    x_tr["campaign_id"] = tr["campaign_id"].values
    x_val["campaign_id"] = val["campaign_id"].values
    return x_tr, x_val


def df2result(df, metric_col, param_cols, param_types=None):
    """Converts dataframe with metrics and hyperparameters to the OptimizeResults format."""
    if not param_types:
        param_types = [float for _ in param_cols]

    df = _prep_df(df, param_cols, param_types)
    param_space = _convert_to_param_space(df, param_cols, param_types)

    results = OptimizeResult()
    results.x_iters = df[param_cols].values
    results.func_vals = df[metric_col].to_list()
    results.x = results.x_iters[np.argmin(results.func_vals)]
    results.fun = np.min(results.func_vals)
    results.space = param_space
    return results


def hpbandster2skopt(results):
    """Converts hpbandster results to scipy OptimizeResult."""
    results = results.get_pandas_dataframe()
    params, loss = results
    params.drop(columns="budget", index=1, inplace=True)
    results_ = params.copy()
    results_["target"] = loss["loss"]
    return df2result(results_, metric_col="target", param_cols=params.columns)


class TrainEvalWorker(Worker):
    """Optimization Worker."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_tr, self.y_tr, self.x_val, self.y_val = load_data("val")
        self.x_tr, self.x_val = map_campign_id(self.x_tr, self.x_val, "val")
        self.x_tr = get_rank_features(self.x_tr)
        self.x_val = get_rank_features(self.x_val)
        # self.y_tr = self.y_tr[self.x_tr.coupon_id != 8]
        # self.x_tr = self.x_tr.loc[self.x_tr.coupon_id != 8]

        self.feats = [
            f
            for i, f in enumerate(self.x_tr.columns)
            if ("coupon_details" not in f)
            and ("common_repeats" not in f)
            and (
                f not in ["campaign_id", "customer_campaign_count", "redemption_status"]
            )
        ]

    def compute(self, config, budget, working_directory, *args, **kwargs):
        lgb_params = {k: v for k, v in config.items() if k not in self.feats}
        # feats_params = {k: v for k, v in config.items() if k in self.feats}
        # use_feats = [f for f, v in feats_params.items() if v == 1]
        lgb_params = {**lgb_params, **STATIC_PARAMS}
        train_data = lgb.Dataset(self.x_tr[self.feats], label=self.y_tr)
        valid_data = lgb.Dataset(
            self.x_val[self.feats], label=self.y_val, reference=train_data
        )

        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=10000,
            early_stopping_rounds=500,
            valid_sets=[valid_data],
            valid_names=["valid"],
            verbose_eval=0,
        )

        score = model.best_score["valid"]["auc"]
        print("Score ->", score)
        loss = -1 * score
        return {"loss": loss, "info": {"auxiliary_stuff": "worked"}}

    def get_configspace(self):
        cs = CS.ConfigurationSpace()
        learning_rate = CSH.UniformFloatHyperparameter(
            "learning_rate", lower=0.003, upper=0.005, default_value=0.004, log=False
        )
        num_leaves = CSH.UniformIntegerHyperparameter(
            "num_leaves", lower=3, upper=4, default_value=3, log=False
        )
        min_data_in_leaf = CSH.UniformIntegerHyperparameter(
            "min_data_in_leaf", lower=400, upper=1000, default_value=700, log=False
        )
        feature_fraction = CSH.UniformFloatHyperparameter(
            "feature_fraction", lower=0.1, upper=0.9, default_value=0.45, log=False
        )
        subsample = CSH.UniformFloatHyperparameter(
            "subsample", lower=0.5, upper=1.0, default_value=0.8, log=False
        )
        l1 = CSH.UniformFloatHyperparameter(
            "lambda_l1", lower=1e-12, upper=10.0, default_value=1.0, log=True
        )
        l2 = CSH.UniformFloatHyperparameter(
            "lambda_l2", lower=1e-12, upper=10.0, default_value=1.0, log=True
        )
        seed = CSH.UniformIntegerHyperparameter(
            "seed", lower=1, upper=10000, default_value=7861
        )
        # feats_flag = [
        #    CSH.UniformIntegerHyperparameter(feat, lower=0, upper=1, default_value=1)
        #    for feat in self.feats
        # ]
        cs.add_hyperparameters(
            [
                learning_rate,
                num_leaves,
                min_data_in_leaf,
                feature_fraction,
                subsample,
                l1,
                l2,
                seed,
            ]
        )
        return cs


def _prep_df(df, param_cols, param_types):
    for col, col_type in zip(param_cols, param_types):
        df[col] = df[col].astype(col_type)
    return df


def _convert_to_param_space(df, param_cols, param_types):
    dimensions = []
    for colname, col_type in zip(param_cols, param_types):
        if col_type == str:
            dimensions.append(
                skopt.space.Categorical(categories=df[colname].unique(), name=colname)
            )
        elif col_type == float:
            low, high = df[colname].min(), df[colname].max()
            dimensions.append(skopt.space.Real(low, high, name=colname))
        else:
            raise NotImplementedError
    skopt_space = skopt.Space(dimensions)
    return skopt_space


if __name__ == "__main__":
    NS = hpns.NameServer(
        run_id=RUN_ID, host=HOST, port=PORT, working_directory=WORKING_DIRECTORY
    )
    ns_host, ns_port = NS.start()

    # Start local worker
    worker = TrainEvalWorker(run_id=RUN_ID, nameserver=ns_host, nameserver_port=ns_port)
    worker.run(background=True)
    result_logger = hpres.json_result_logger(
        directory="data/hpbandster/{}".format(RUN_ID), overwrite=False
    )
    optim = BOHB(
        configspace=worker.get_configspace(),
        run_id=RUN_ID,
        nameserver=ns_host,
        nameserver_port=ns_port,
        result_logger=result_logger,
        eta=HPO_PARAMS["eta"],
        min_budget=HPO_PARAMS["min_budget"],
        max_budget=HPO_PARAMS["max_budget"],
        num_samples=HPO_PARAMS["num_samples"],
        top_n_percent=HPO_PARAMS["top_n_percent"],
        min_bandwidth=HPO_PARAMS["min_bandwidth"],
        bandwidth_factor=HPO_PARAMS["bandwidth_factor"],
    )
    study = optim.run(n_iterations=HPO_PARAMS["n_calls"])

    results = hpbandster2skopt(study)

    best_auc = -1.0 * results.fun
    best_params = results.x
    print(results)
    # log metrics
    print("Best Validation AUC: {}".format(best_auc))
    print("Best Params: {}".format(best_params))

    optim.shutdown(shutdown_workers=True)
    NS.shutdown()
