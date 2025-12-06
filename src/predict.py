import pandas as pd
import xgboost as xgb
from .config import TEST_FEATURES_PATH, TRAIN_FEATURES_PATH, TEST_RAW_PATH, MODEL_PATH
from .models import FEATURE_EXCLUDE


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes
    return df


def main():
    # 1) Load train features to get exact feature list
    train_feat = pd.read_parquet(TRAIN_FEATURES_PATH)
    train_feat = _encode_categoricals(train_feat)
    train_feature_cols = [c for c in train_feat.columns if c not in FEATURE_EXCLUDE]

    # 2) Load processed test features (no id here)
    test_feat = pd.read_parquet(TEST_FEATURES_PATH)

    # Ensure all training feature columns exist in test; create missing as 0
    for col in train_feature_cols:
        if col not in test_feat.columns:
            test_feat[col] = 0

    # Drop any extra columns
    test_feat = test_feat[train_feature_cols]

    # 3) Encode categoricals in test
    test_feat = _encode_categoricals(test_feat)

    # 4) Build DMatrix with same feature order
    dtest = xgb.DMatrix(test_feat[train_feature_cols])

    # 5) Load ids from raw test.csv
    test_raw = pd.read_csv(TEST_RAW_PATH)
    ids = test_raw["id"]

    # 6) Load model and predict
    model = xgb.Booster()
    model.load_model(MODEL_PATH)

    preds = model.predict(dtest)

    submission = pd.DataFrame(
        {
            "id": ids,
            "sales": preds,
        }
    )
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")


if __name__ == "__main__":
    main()
