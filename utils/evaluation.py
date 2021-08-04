import numpy as np
import pandas as pd
from typing import Union
from tqdm.notebook import tqdm


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
        key_columns = id_columns = train.loc[
            :, ~train.columns.str.startswith("d_")
        ].columns.tolist()
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

    def rmsse(self, pred: pd.DataFrame, level: int) -> pd.Series:
        test_total_quantities = getattr(self, f"level-{level}_test_total_quantities")
        score = ((test_total_quantities - pred) ** 2).mean(axis=1)
        scale = getattr(self, f"level-{level}_scale")
        return (score / scale).map(np.sqrt)

    def score(self, preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.test[self.test_target_columns].shape == preds.shape

        if isinstance(preds, np.ndarray):
            preds = pd.DataFrame(preds, columns=self.test_target_columns)

        preds = pd.concat([self.test[self.key_columns], preds], axis=1, sort=False)

        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            level_scores = self.rmsse(
                preds.groupby(group_id)[self.test_target_columns].sum(), i + 1
            )
            weight = getattr(self, f"level-{i + 1}_weight")
            level_scores = pd.concat([weight, level_scores], axis=1, sort=False).prod(
                axis=1
            )
            all_scores.append(level_scores.sum())

        return np.mean(all_scores)
