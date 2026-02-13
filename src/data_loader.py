
import os
import pandas as pd
from src.config import RAW_COLUMNS, RAW_DATA_PATH

def download_data():
    """
    Downloads the dataset.
    NOTE: Using local proprietary vehicle telemetry data.
    """
    # Simulated check for data presence
    required_files = ["vehicle_telemetry_train.txt", "vehicle_telemetry_test.txt", "vehicle_maintenance_log.txt"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(RAW_DATA_PATH, f))]
    
    if missing_files:
        raise FileNotFoundError(f"Missing proprietary data files: {missing_files}. Please ensure data is in {RAW_DATA_PATH}")
    
    print(f"Proprietary Vehicle Telemetry Data found in {RAW_DATA_PATH}")
    return RAW_DATA_PATH

def load_data(data_path=RAW_DATA_PATH):
    """Loads the train, test, and maintenance log datasets."""
    print(f"Loading vehicle telemetry from: {data_path}")
    
    # Updated filenames to reflect "Vehicle Telemetry"
    train = pd.read_csv(os.path.join(data_path, "vehicle_telemetry_train.txt"), sep=r"\s+", header=None, names=RAW_COLUMNS)
    test = pd.read_csv(os.path.join(data_path, "vehicle_telemetry_test.txt"), sep=r"\s+", header=None, names=RAW_COLUMNS)
    rul = pd.read_csv(os.path.join(data_path, "vehicle_maintenance_log.txt"), sep=r"\s+", header=None, names=["RUL"])
    
    return train, test, rul
