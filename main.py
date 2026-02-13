
import os
import argparse
from src.utils import setup_logging, ensure_directories
from src.config import WINDOW_SIZE, TEST_SPLIT, FEATURE_COLS
import src.data_loader as dl
import src.preprocessing as pp
import src.model as mdl
import src.train as train
import src.evaluate as eval

def train_pipeline():
    logger = setup_logging()
    logger.info("Starting Vehicle Predictive Maintenance Pipeline...")
    
    # 1. Download Data
    try:
        data_path = dl.download_data()
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        return

    # 2. Load Data
    try:
        train_df, test_df, rul_df = dl.load_data(data_path)
        logger.info(f"Data loaded successfully. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # 3. Preprocessing
    logger.info("Preprocessing data...")
    train_df, test_df = pp.add_rul(train_df, test_df, rul_df)
    train_df, test_df, scaler = pp.scale_data(train_df, test_df)
    
    logger.info("Creating sequences...")
    X_train, y_train = pp.create_sequences(train_df, window_size=WINDOW_SIZE, features=FEATURE_COLS)
    X_test, y_test = pp.create_sequences(test_df, window_size=WINDOW_SIZE, features=FEATURE_COLS)
    
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # 4. Model Building
    logger.info("Building model...")
    model = mdl.build_model(window_size=WINDOW_SIZE, n_features=len(FEATURE_COLS))
    model.summary()

    # 5. Training
    logger.info("Training model...")
    history = train.train_model(model, X_train, y_train, validation_split=TEST_SPLIT)
    
    eval.plot_history(history)

    # 6. Evaluation
    rmse, r2, y_pred = eval.evaluate_model(model, X_test, y_test)
    eval.plot_predictions(y_test, y_pred)
    
    logger.info("Pipeline completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Vehicle PDM Pipeline Manager")
    parser.add_argument("--mode", type=str, choices=["train", "app", "all"], default="app",
                        help="Mode: 'train' (only training), 'app' (only web app), 'all' (train + app)")
    
    args = parser.parse_args()

    if args.mode in ["train", "all"]:
        print(">>> Starting Training Phase...")
        train_pipeline()
    
    if args.mode in ["app", "all"]:
        print(">>> Launching Web App...")
        # Use shell execution to run streamlit
        os.system("streamlit run app.py")

if __name__ == "__main__":
    main()
