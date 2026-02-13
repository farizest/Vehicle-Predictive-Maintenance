
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from src.config import RUL_CAP, FEATURE_COLS, SCALER_PATH

def add_rul(train_df, test_df, rul_df):
    """Calculates and adds RUL (Remaining Useful Life) to datasets."""
    # Train RUL
    rul_train = train_df.groupby("unit")["cycle"].max().reset_index()
    rul_train.columns = ["unit", "max_cycle"]
    train_df = train_df.merge(rul_train, on="unit", how="left")
    train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]
    train_df.drop("max_cycle", axis=1, inplace=True)

    # Test RUL
    rul_test = test_df.groupby("unit")["cycle"].max().reset_index()
    rul_test.columns = ["unit", "last_cycle"]
    rul_test = rul_test.merge(rul_df, left_index=True, right_index=True)
    rul_test["max_cycle"] = rul_test["last_cycle"] + rul_test["RUL"]
    test_df = test_df.merge(rul_test[["unit", "max_cycle"]], on="unit", how="left")
    test_df["RUL"] = test_df["max_cycle"] - test_df["cycle"]
    test_df.drop("max_cycle", axis=1, inplace=True)
    
    # Cap RUL
    train_df["RUL"] = np.where(train_df["RUL"] > RUL_CAP, RUL_CAP, train_df["RUL"])
    test_df["RUL"] = np.where(test_df["RUL"] > RUL_CAP, RUL_CAP, test_df["RUL"])
    
    return train_df, test_df

def scale_data(train_df, test_df, save_scaler=True):
    """Scales feature columns using MinMaxScaler."""
    scaler = MinMaxScaler()
    
    # Scale features
    train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])
    test_df[FEATURE_COLS] = scaler.transform(test_df[FEATURE_COLS])
    
    if save_scaler:
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"Scaler saved to {SCALER_PATH}")
        
    return train_df, test_df, scaler

def create_sequences(df, window_size=30, features=FEATURE_COLS):
    """Creates sliding window sequences for LSTM."""
    X, y = [], []
    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit]
        unit_features = unit_df[features].values
        unit_rul = unit_df["RUL"].values
        
        for i in range(len(unit_df) - window_size + 1):
            X.append(unit_features[i:i+window_size])
            # Predict RUL at the end of the window
            y.append(unit_rul[i+window_size-1])
            
    return np.array(X), np.array(y)
