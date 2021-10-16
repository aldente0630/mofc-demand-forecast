import numpy as np
import pandas as pd
from typing import Union
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm.notebook import tqdm


def calc_eval_metric(y_true, y_pred):
    eval_metrics = dict()

    for index in y_true.index:
        eval_metrics_by_id = dict()

        eval_metrics_by_id["mae"] = mean_absolute_error(
            y_true.loc[index, :], y_pred.loc[index, :]
        )
        eval_metrics_by_id["rmse"] = np.sqrt(
            mean_squared_error(y_true.loc[index, :], y_pred.loc[index, :])
        )
        eval_metrics_by_id["smape"] = mean_absolute_percentage_error(
            y_true.loc[index, :], y_pred.loc[index, :], is_symmetric=True
        )
        eval_metrics_by_id["mase"] = mean_absolute_scaled_error(
            y_true.loc[index, :], y_pred.loc[index, :]
        )

        eval_metrics[index] = eval_metrics_by_id

    return pd.DataFrame(eval_metrics).T


def mean_absolute_percentage_error(y_true, y_pred, is_symmetric=False):
    if is_symmetric:
        return np.nanmean(2 * np.abs((y_true - y_pred) / (y_true + y_pred)))
    else:
        return np.nanmean(p.abs((y_true - y_pred) / y_true))


def mean_absolute_scaled_error(y_true, y_pred, seasonality=1):
    naive_forecast = y_true[:-seasonality]
    denominator = mean_absolute_error(y_true[seasonality:], naive_forecast)
    return (
        mean_absolute_error(y_true, y_pred) / denominator
        if denominator > 0.0
        else np.nan
    )


class WRMSSEEvaluator(object):
    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        calendar: pd.DataFrame,
        selling_prices: pd.DataFrame,
        test_steps: int,
    ):
        train = df_train.copy()
        test = df_test.copy()

        target = train.loc[:, train.columns.str.startswith("d_")]
        train_target_columns = target.columns.tolist()
        weight_columns = target.iloc[:, -test_steps:].columns.tolist()
        train["all_id"] = 0
        key_columns = train.loc[:, ~train.columns.str.startswith("d_")].columns.tolist()
        test_target_columns = test.loc[
            :, test.columns.str.startswith("d_")
        ].columns.tolist()

        if not all([column in test.columns for column in key_columns]):
            test = pd.concat([train[key_columns], test], axis=1, sort=False)

        self.train = train
        self.test = test
        self.calendar = calendar
        self.selling_prices = selling_prices
        self.weight_columns = weight_columns
        self.key_columns = key_columns
        self.test_target_columns = test_target_columns

        sales_weights = self.get_sales_weight()

        self.group_ids = (
            "all_id",
            "state_id",
            "store_id",
            "cat_id",
            "dept_id",
            ["state_id", "cat_id"],
            ["state_id", "dept_id"],
            ["store_id", "cat_id"],
            ["store_id", "dept_id"],
            "item_id",
            ["item_id", "state_id"],
            ["item_id", "store_id"],
        )

        for i, group_id in enumerate(tqdm(self.group_ids)):
            train_total_quantities = train.groupby(group_id)[train_target_columns].sum()
            scale = []
            for _, row in train_total_quantities.iterrows():
                series = row.values[np.argmax(row.values != 0) :]
                scale.append(((series[1:] - series[:-1]) ** 2).mean())
            setattr(self, f"level-{i + 1}_scale", np.array(scale))
            setattr(
                self, f"level-{i + 1}_train_total_quantities", train_total_quantities
            )
            setattr(
                self,
                f"level-{i + 1}_test_total_quantities",
                test.groupby(group_id)[test_target_columns].sum(),
            )
            level_weight = (
                sales_weights.groupby(group_id)[weight_columns].sum().sum(axis=1)
            )
            setattr(self, f"level-{i + 1}_weight", level_weight / level_weight.sum())

    def get_sales_weight(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index("d")["wm_yr_wk"].to_dict()
        sales_weights = self.train[
            ["item_id", "store_id"] + self.weight_columns
        ].set_index(["item_id", "store_id"])
        sales_weights = (
            sales_weights.stack()
            .reset_index()
            .rename(columns={"level_2": "d", 0: "value"})
        )
        sales_weights["wm_yr_wk"] = sales_weights["d"].map(day_to_week)

        sales_weights = sales_weights.merge(
            self.selling_prices, how="left", on=["item_id", "store_id", "wm_yr_wk"]
        )
        sales_weights["value"] = sales_weights["value"] * sales_weights["sell_price"]
        sales_weights = sales_weights.set_index(["item_id", "store_id", "d"]).unstack(
            level=2
        )["value"]
        sales_weights = sales_weights.loc[
            zip(self.train["item_id"], self.train["store_id"]), :
        ].reset_index(drop=True)
        sales_weights = pd.concat(
            [self.train[self.key_columns], sales_weights], axis=1, sort=False
        )
        return sales_weights

    def rmsse(self, prediction: pd.DataFrame, level: int) -> pd.Series:
        test_total_quantities = getattr(self, f"level-{level}_test_total_quantities")
        score = ((test_total_quantities - prediction) ** 2).mean(axis=1)
        scale = getattr(self, f"level-{level}_scale")
        return (score / scale).map(np.sqrt)

    def score(self, predictions: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.test[self.test_target_columns].shape == predictions.shape

        if isinstance(predictions, np.ndarray):
            predictions = pd.DataFrame(predictions, columns=self.test_target_columns)

        predictions = pd.concat(
            [self.test[self.key_columns], predictions], axis=1, sort=False
        )

        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            level_scores = self.rmsse(
                predictions.groupby(group_id)[self.test_target_columns].sum(), i + 1
            )
            weight = getattr(self, f"level-{i + 1}_weight")
            level_scores = pd.concat([weight, level_scores], axis=1, sort=False).prod(
                axis=1
            )
            all_scores.append(level_scores.sum())

        return np.mean(all_scores)
