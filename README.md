# AutoDoc: AI-Powered Vehicle Predictive Maintenance

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

**AutoDoc** is an intelligent predictive maintenance system designed to estimate the **Remaining Useful Life (RUL)** of vehicle engines. It leverages a **BiLSTM (Bidirectional LSTM)** deep learning model with a custom **Attention Mechanism** to analyze temporal sensor data and predict failures before they happen.

The system features a **GenAI Mechanic** (powered by Google Gemini) that translates complex sensor diagnostics into simple, actionable advice for drivers.

## 🚀 Features

-   **Predictive AI**: BiLSTM + Attention model for accurate RUL estimation.
-   **Live Telemetry Simulation**: interactively flows through real-world test flight cycles.
-   **AI Mechanic**: Generates human-readable diagnostic reports using LLMs (Gemini).
    -   Identify *What* is wrong.
    -   Pinpoint *Where* the component is.
    -   Suggest *How* to fix it.
-   **Interactive Dashboard**: Built with Streamlit for real-time visualization.
-   **Flexible Pipeline**: Command-line interface to Train, Evaluate, or Run the App.

## 📂 Project Structure

```
vehicle_pdm/
├── data/                  # Vehicle Telemetry Data (Train/Test)
├── models/                # Saved models (.h5) and scalers (.pkl)
├── src/                   # Source code modules
│   ├── config.py          # Configuration parameters
│   ├── data_loader.py     # Data ingestion
│   ├── preprocessing.py   # Feature engineering & sequence generation
│   ├── model.py           # Attention-BiLSTM Architecture
│   ├── train.py           # Training loop
│   └── evaluate.py        # Evaluation metrics & plotting
├── app.py                 # Streamlit Web Application
├── main.py                # Main execution entry point
├── requirements.txt       # Dependencies
└── .env                   # API Keys (Not shared)
```

## 🛠️ Setup & Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/vehicle_pdm.git
    cd vehicle_pdm
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Keys**:
    -   Create a file named `.env` in the root directory.
    -   Add your Google Gemini API key:
        ```env
        GEMINI_KEY=your_api_key_here
        ```

## 🏃 Usage

You can run the project in different modes using `main.py`:

### 1. Run the Web Dashboard (Recommended)
Launch the interactive application to explore the model and AI mechanic.
```bash
python main.py --mode app
```

### 2. Train the Model
Retrain the model from scratch using the data in `data/`.
```bash
python main.py --mode train
```

### 3. Full Pipeline (Train + App)
Train the model and immediately launch the app.
```bash
python main.py --mode all
```

## 🧠 Model Details

-   **Input**: Time-series sensor data (Sequence Length: 30 cycles).
-   **Features**: 17 engine sensor readings (Temperature, Pressure, RPM, etc.).
-   **Architecture**:
    -   Input Layer
    -   Bidirectional LSTM (64 units)
    -   Attention Layer (Focuses on critical time steps)
    -   Dense Layers (Output RUL)
-   **Performance**: Minimizes RMSE (Root Mean Squared Error) on test data.

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
