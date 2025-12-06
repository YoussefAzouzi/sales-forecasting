import pandas as pd
from typing import Tuple
from .config import (
    TRAIN_RAW_PATH, TEST_RAW_PATH,
    STORES_PATH, OIL_PATH, HOLIDAYS_PATH, TRANSACTIONS_PATH,
    TRAIN_MERGED_PATH,
)

def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                             pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_RAW_PATH, parse_dates=["date"])
    test = pd.read_csv(TEST_RAW_PATH, parse_dates=["date"])
    stores = pd.read_csv(STORES_PATH)
    oil = pd.read_csv(OIL_PATH, parse_dates=["date"])
    holidays = pd.read_csv(HOLIDAYS_PATH, parse_dates=["date"])
    transactions = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["date"])
    return train, test, stores, oil, holidays, transactions

def merge_external_data(
    df: pd.DataFrame,
    stores: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    # Merge store info
    df = df.merge(stores, on="store_nbr", how="left")

    # Oil price: forward fill
    oil_sorted = oil.sort_values("date")
    oil_sorted["dcoilwtico"] = oil_sorted["dcoilwtico"].ffill()
    df = df.merge(oil_sorted, on="date", how="left")

    # Holidays: keep useful columns and aggregate to a single flag
    hol = holidays.copy()
    hol["is_holiday"] = 1
    hol = hol[["date", "is_holiday"]].drop_duplicates()
    df = df.merge(hol, on="date", how="left")
    df["is_holiday"] = df["is_holiday"].fillna(0).astype("int8")

    # Transactions per store/date
    df = df.merge(transactions, on=["date", "store_nbr"], how="left")
    df["transactions"] = df["transactions"].fillna(0)

    return df

def build_merged_train_test() -> None:
    train, test, stores, oil, holidays, transactions = load_raw_data()

    train_merged = merge_external_data(train, stores, oil, holidays, transactions)
    test_merged = merge_external_data(test, stores, oil, holidays, transactions)

    # Save only train_merged here; test merged will be regenerated in features.py
    train_merged.to_parquet(TRAIN_MERGED_PATH, index=False)

if __name__ == "__main__":
    build_merged_train_test()
