from .data_loader import build_merged_train_test
from .features import create_train_features, create_test_features
from .models import train_xgb_model

def main():
    print("Building merged train data...")
    build_merged_train_test()

    print("Creating train features...")
    create_train_features()

    print("Creating test features...")
    create_test_features()

    print("Training XGBoost model...")
    train_xgb_model()

    print("Training complete.")

if __name__ == "__main__":
    main()
