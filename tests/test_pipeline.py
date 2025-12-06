import pandas as pd
from pathlib import Path
from src import config
from src.data_loader import build_merged_train_test
from src.features import create_train_features, create_test_features
from src.models import train_xgb_model

def test_end_to_end():
    # Run data + feature + model pipeline
    build_merged_train_test()
    create_train_features()
    create_test_features()
    train_xgb_model(num_boost_round=10, early_stopping_rounds=5)

    assert Path(config.TRAIN_FEATURES_PATH).exists()
    assert Path(config.TEST_FEATURES_PATH).exists()
    assert Path(config.MODEL_PATH).exists()

    df_train = pd.read_parquet(config.TRAIN_FEATURES_PATH)
    assert "sales" in df_train.columns
