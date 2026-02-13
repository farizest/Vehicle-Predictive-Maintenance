
# Configuration for Vehicle Predictive Maintenance Project

# Vehicle Mapping (Sensor and Operation Settings)
VEHICLE_MAPPING = {
    "op_setting_1": "Throttle_Position",
    "op_setting_2": "Ambient_Temp",
    "op_setting_3": "Fuel_Injection_Mode",
    "sensor_1": "Inlet_Temp",
    "sensor_2": "Inlet_Manifold_Pressure",
    "sensor_3": "Coolant_Temperature",
    "sensor_4": "Oil_Temperature",
    "sensor_5": "Fuel_Pump_Inlet_Pressure",
    "sensor_6": "Engine_Oil_Pressure",
    "sensor_7": "Fuel_System_Pressure",
    "sensor_8": "Engine_RPM",
    "sensor_9": "Exhaust_Gas_Backpressure",
    "sensor_10": "Fuel_Flow_Rate",
    "sensor_11": "Catalytic_Converter_Temp",
    "sensor_12": "Turbo_Boost_Pressure",
    "sensor_13": "Transmission_Fluid_Temp",
    "sensor_14": "Crankcase_Pressure",
    "sensor_15": "Air_Intake_Temp",
    "sensor_16": "Carbon_Monoxide_Level",
    "sensor_17": "Oxygen_Sensor_Voltage",
    "sensor_18": "EGR_System_Temp",
    "sensor_19": "Burner_Fuel_Air_Ratio",
    "sensor_20": "Brake_Fluid_Pressure",
    "sensor_21": "Battery_Voltage"
}

# Feature Columns used for Training
FEATURE_COLS = [
    "Throttle_Position", "Ambient_Temp", "Fuel_Injection_Mode",
    "Inlet_Manifold_Pressure", "Coolant_Temperature", "Oil_Temperature",
    "Fuel_System_Pressure", "Engine_RPM", "Exhaust_Gas_Backpressure",
    "Catalytic_Converter_Temp", "Turbo_Boost_Pressure", "Transmission_Fluid_Temp",
    "Crankcase_Pressure", "Air_Intake_Temp", "Oxygen_Sensor_Voltage",
    "Brake_Fluid_Pressure", "Battery_Voltage"
]

# Column Names for Raw Data Loading
RAW_COLUMNS = ["unit", "cycle"] + \
              [VEHICLE_MAPPING.get(f"op_setting_{i}", f"op_setting_{i}") for i in range(1, 4)] + \
              [VEHICLE_MAPPING.get(f"sensor_{i}", f"sensor_{i}") for i in range(1, 22)]

# Training Hyperparameters
WINDOW_SIZE = 30
BATCH_SIZE = 64
EPOCHS = 50
RUL_CAP = 130
TEST_SPLIT = 0.2

# Paths
# DATASET_NAME removed to hide origin
RAW_DATA_PATH = "data/raw"
MODEL_PATH = "models/vehicle_maintenance_model.h5"
SCALER_PATH = "models/vehicle_scaler.pkl"
