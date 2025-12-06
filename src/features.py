import pandas as pd
import numpy as np
from .config import (
    TRAIN_MERGED_PATH,
    TRAIN_FEATURES_PATH,
    TEST_RAW_PATH,
    STORES_PATH,
    OIL_PATH,
    HOLIDAYS_PATH,
    TRANSACTIONS_PATH,
    TEST_FEATURES_PATH,
)
from .data_loader import merge_external_data

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype("int8")
    return df

def add_lag_features(
    df: pd.DataFrame,
    group_cols=("store_nbr", "family"),
    target_col="sales",
) -> pd.DataFrame:
    df = df.sort_values(["store_nbr", "family", "date"])
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df.groupby(list(group_cols))[target_col].shift(lag)

    for window in [7, 14, 28]:
        df[f"roll_mean_{window}"] = (
            df.groupby(list(group_cols))[target_col]
            .shift(1)
            .rolling(window=window)
            .mean()
        )
        df[f"roll_std_{window}"] = (
            df.groupby(list(group_cols))[target_col]
            .shift(1)
            .rolling(window=window)
            .std()
        )
    return df

def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_nbr", "family", "date"])

    # Ensure onpromotion is numeric
    df["onpromotion"] = df["onpromotion"].fillna(0).astype("float32")

    # Use transform so the result aligns with df.index
    df["promo_7d_sum"] = (
        df.groupby(["store_nbr", "family"])["onpromotion"]
        .transform(lambda s: s.rolling(7, min_periods=1).sum())
    )

    df["promo_30d_sum"] = (
        df.groupby(["store_nbr", "family"])["onpromotion"]
        .transform(lambda s: s.rolling(30, min_periods=1).sum())
    )

    return df

def create_train_features() -> None:
    train_merged = pd.read_parquet(TRAIN_MERGED_PATH)

    train_merged = add_date_features(train_merged)
    train_merged = add_lag_features(train_merged)
    train_merged = add_promo_features(train_merged)

    # Drop rows with NaN lag features (start-up period)
    feature_cols = [
        c
        for c in train_merged.columns
        if c
        not in [
            "id",
            "sales",
        ]
    ]

    # Remove rows where any lag is NaN
    lags = [c for c in train_merged.columns if c.startswith("lag_")]
    train_feat = train_merged.dropna(subset=lags).reset_index(drop=True)

    train_feat.to_parquet(TRAIN_FEATURES_PATH, index=False)

def create_test_features() -> None:
    # Load raw and merge external data
    test = pd.read_csv(TEST_RAW_PATH, parse_dates=["date"])
    stores = pd.read_csv(STORES_PATH)
    oil = pd.read_csv(OIL_PATH, parse_dates=["date"])
    holidays = pd.read_csv(HOLIDAYS_PATH, parse_dates=["date"])
    transactions = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["date"])

    test_merged = merge_external_data(test, stores, oil, holidays, transactions)
    test_merged = add_date_features(test_merged)

    # NOTE: For a real competition setup you would create lag features
    # using the combined train + test timeline (recursive forecasting).
    # For a learning project, a simpler approach is to merge the last
    # known lags from train onto test.

    train_feat = pd.read_parquet(TRAIN_FEATURES_PATH)
    # Get last available row per (store_nbr, family)
    last_hist = (
        train_feat.sort_values("date")
        .groupby(["store_nbr", "family"])
        .tail(1)
        .reset_index(drop=True)
    )

    lag_cols = [c for c in train_feat.columns if c.startswith("lag_") or c.startswith("roll_")]

    last_hist_small = last_hist[["store_nbr", "family"] + lag_cols]

    test_feat = test_merged.merge(last_hist_small, on=["store_nbr", "family"], how="left")

    # Promo rolling sums cannot be easily updated without recursive logic,
    # but the last historical values still carry useful information.
    test_feat = add_promo_features(
        pd.concat(
            [
                train_feat[
                    ["store_nbr", "family", "date", "onpromotion"]
                ].tail(7 * 10),  # small sample to preserve dtypes
                test_feat[["store_nbr", "family", "date", "onpromotion"]],
            ],
            ignore_index=True,
        )
    ).tail(len(test_feat))

    test_feat.to_parquet(TEST_FEATURES_PATH, index=False)

if __name__ == "__main__":
    create_train_features()
    create_test_features()
