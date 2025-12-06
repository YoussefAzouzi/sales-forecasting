import xgboost as xgb
import pandas as pd
from typing import Tuple, List
from .config import TRAIN_FEATURES_PATH, MODEL_PATH

FEATURE_EXCLUDE = [
    "id",
    "date",
    "sales",
]

def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    # Find non-numeric columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes
    return df

def load_train_matrix(
    valid_start_date: str = "2017-07-01",
) -> Tuple[xgb.DMatrix, xgb.DMatrix, List[str]]:
    df = pd.read_parquet(TRAIN_FEATURES_PATH)

    # Encode any remaining object columns
    df = _encode_categoricals(df)

    feature_cols = [c for c in df.columns if c not in FEATURE_EXCLUDE]
    train_df = df[df["date"] < valid_start_date]
    valid_df = df[df["date"] >= valid_start_date]

    X_train = train_df[feature_cols]
    y_train = train_df["sales"]
    X_valid = valid_df[feature_cols]
    y_valid = valid_df["sales"]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    return dtrain, dvalid, feature_cols

def train_xgb_model(
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
) -> None:
    dtrain, dvalid, feature_cols = load_train_matrix()

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    evals = [(dtrain, "train"), (dvalid, "valid")]
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50,
    )

    model.save_model(MODEL_PATH)

def load_trained_model() -> xgb.Booster:
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    return model
