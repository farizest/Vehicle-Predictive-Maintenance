from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from tensorflow.keras.losses import MeanSquaredError

import src.data_loader as dl
from src.config import FEATURE_COLS, MODEL_PATH, SCALER_PATH, WINDOW_SIZE
from src.model import Attention

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None


ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))


class PredictResponse(BaseModel):
    prediction_km: float
    status: Literal["HEALTHY", "WARNING", "CRITICAL"]
    sensors: dict[str, float]
    xai: list[dict[str, Any]]
    scenario: dict[str, Any] | None = None


class ManualPredictRequest(BaseModel):
    sensors: dict[str, float] = Field(default_factory=dict)
    seed: int | None = None


class DiagnoseRequest(BaseModel):
    prediction_km: float
    sensors: dict[str, float]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    prediction_km: float
    worst_sensor: str


def _status_from_prediction(prediction_km: float) -> Literal["HEALTHY", "WARNING", "CRITICAL"]:
    if prediction_km > 75:
        return "HEALTHY"
    if prediction_km > 30:
        return "WARNING"
    return "CRITICAL"


@lru_cache(maxsize=1)
def load_assets() -> tuple[tf.keras.Model, Any, pd.DataFrame]:
    custom_objects = {"Attention": Attention, "mse": MeanSquaredError()}

    model_path = ROOT / MODEL_PATH
    scaler_path = ROOT / SCALER_PATH

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Run training first.")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Run training first.")

    model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
    scaler = joblib.load(str(scaler_path))

    _, test_df, _ = dl.load_data()
    test_df = test_df.copy()
    test_df[FEATURE_COLS] = scaler.transform(test_df[FEATURE_COLS])
    return model, scaler, test_df


@tf.function
def get_gradients(model, inputs):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        prediction = model(inputs)
    return tape.gradient(prediction, inputs)

def calculate_xai(model: tf.keras.Model, X: np.ndarray) -> list[dict[str, Any]]:
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    baseline = tf.zeros_like(X_tensor)
    steps = 50
    
    alphas = tf.linspace(start=0.0, stop=1.0, num=steps+1)
    alphas = tf.reshape(alphas, (steps+1, 1, 1, 1))
    
    delta = X_tensor - baseline
    interpolated_path = baseline + alphas * delta
    
    grads = []
    for i in range(steps + 1):
        grads.append(get_gradients(model, interpolated_path[i]))
        
    grads = tf.stack(grads)
    avg_grads = tf.reduce_mean(grads, axis=0)
    
    integrated_gradients = avg_grads * delta
    feature_importance = tf.reduce_sum(integrated_gradients, axis=1)[0].numpy()
    
    rows = []
    for i, sensor_name in enumerate(FEATURE_COLS):
        # The mathematical attribution naturally defines the sensor's health impact
        rows.append({
            "component": sensor_name,
            "health_impact": float(feature_importance[i]),
            "value": float(X[0, -1, i])
        })

    # Sort by absolute magnitude (highest impact first, whether negative or positive)
    rows.sort(key=lambda r: abs(r["health_impact"]), reverse=True)
    
    # Return only the top 5 most influential sensors
    return rows[:5]


def _sequence_for_unit_cycle(test_df: pd.DataFrame, unit: int, cycle: int) -> np.ndarray:
    unit_data = test_df[test_df["unit"] == unit]
    if unit_data.empty:
        raise ValueError(f"Unknown unit: {unit}")
    max_cycle = int(unit_data["cycle"].max())
    if cycle < WINDOW_SIZE or cycle > max_cycle:
        raise ValueError(f"cycle must be in [{WINDOW_SIZE}, {max_cycle}] for unit={unit}")

    seq_df = unit_data[(unit_data["cycle"] > cycle - WINDOW_SIZE) & (unit_data["cycle"] <= cycle)]
    if len(seq_df) != WINDOW_SIZE:
        raise ValueError("Unable to create a full sequence window.")
    return seq_df[FEATURE_COLS].values.reshape(1, WINDOW_SIZE, len(FEATURE_COLS))


def _random_live_scenario(test_df: pd.DataFrame, seed: int | None = None) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    unit_ids = test_df["unit"].unique()
    unit = int(rng.choice(unit_ids))
    unit_data = test_df[test_df["unit"] == unit]
    max_cycle = int(unit_data["cycle"].max())
    if max_cycle < WINDOW_SIZE:
        # Extremely unlikely; just fall back deterministically
        return {"unit": unit, "cycle": WINDOW_SIZE}
    cycle = int(rng.integers(WINDOW_SIZE, max_cycle + 1))
    return {"unit": unit, "cycle": cycle}


def _gemini_model() -> Any:
    # Force reading from os.environ directly, which is updated by dotenv
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    if genai is None:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


app = FastAPI(title="AutoDoc: Vehicle Predictive Maintenance API", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/features")
def get_features() -> list[str]:
    return FEATURE_COLS


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    return templates.TemplateResponse("index.html", {"request": request, "title": "AutoDoc: Live Telemetry"})

@app.get("/predict/live", response_model=PredictResponse)
def predict_live(
    unit: int | None = Query(default=None),
    cycle: int | None = Query(default=None),
    seed: int | None = Query(default=None),
) -> PredictResponse:
    try:
        model, _scaler, test_df = load_assets()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        if unit is None or cycle is None:
            scenario = _random_live_scenario(test_df, seed=seed)
            unit = scenario["unit"]
            cycle = scenario["cycle"]
        else:
            scenario = {"unit": int(unit), "cycle": int(cycle)}

        X = _sequence_for_unit_cycle(test_df, int(unit), int(cycle))
        prediction = float(model.predict(X, verbose=0).flatten()[0])
        sensors_arr = X[0, -1, :]
        sensors = {FEATURE_COLS[i]: float(sensors_arr[i]) for i in range(len(FEATURE_COLS))}
        status = _status_from_prediction(prediction)
        xai = calculate_xai(model, X)
        return PredictResponse(prediction_km=prediction, status=status, sensors=sensors, xai=xai, scenario=scenario)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/manual", response_model=PredictResponse)
def predict_manual(payload: ManualPredictRequest) -> PredictResponse:
    try:
        model, _scaler, _test_df = load_assets()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    rng = np.random.default_rng(payload.seed)
    X = rng.random((1, WINDOW_SIZE, len(FEATURE_COLS)), dtype=np.float32)

    # Apply manual sensor adjustments
    for sensor_name, val in payload.sensors.items():
        if sensor_name in FEATURE_COLS:
            idx = FEATURE_COLS.index(sensor_name)
            X[0, -1, idx] = float(val)

    prediction = float(model.predict(X, verbose=0).flatten()[0])
    sensors_arr = X[0, -1, :]
    sensors = {FEATURE_COLS[i]: float(sensors_arr[i]) for i in range(len(FEATURE_COLS))}
    status = _status_from_prediction(prediction)
    xai = calculate_xai(model, X)
    return PredictResponse(prediction_km=prediction, status=status, sensors=sensors, xai=xai, scenario={"mode": "manual"})


@app.post("/diagnose")
def diagnose(payload: DiagnoseRequest) -> dict[str, str]:
    model = _gemini_model()
    if model is None:
        return {"markdown": "AI Diagnostics disabled. Set `GEMINI_API_KEY` on the server to enable."}

    status_context = (
        "CRITICAL: Engine breakdown imminent."
        if payload.prediction_km < 30
        else "WARNING: Maintenance advised soon."
        if payload.prediction_km < 75
        else "HEALTHY: Engine operating normally."
    )
    sensors_fmt = {k: f"{float(v):.2f}" for k, v in payload.sensors.items()}

    prompt = f"""
You are AutoDoc, an expert car mechanic AI.
Analysis Context:
- Predicted Remaining Useful Life (RUL): {payload.prediction_km:.0f} km.
- Current Sensor Readings (Normalized 0-1): {sensors_fmt}
- Status Assessment: {status_context}

Task:
Explain the vehicle's health status in simple, non-technical terms for a regular driver.
Focus on the most likely issues based on the RUL and sensor readings (high/low values are suspicious).

Output Format (Markdown):
**1. 🛑 The Issue (What's wrong?)**: [1 sentence]
**2. 📍 The Component (Where is it?)**: [1 sentence]
**3. 🔧 Repair Action (How to fix?)**: [1 sentence]
**4. 💡 Why?**: [1 sentence]

Keep it short, encouraging, and easy to understand. Each output in separate line.
""".strip()

    try:
        text = model.generate_content(prompt).text
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=502, detail=f"AI Explanation unavailable: {e}")
    return {"markdown": text}


@app.post("/chat")
def chat(payload: ChatRequest) -> dict[str, str]:
    model = _gemini_model()
    if model is None:
        return {"reply": "Chatbot disabled. Set `GEMINI_API_KEY` on the server to enable."}

    context = (
        f"You are AutoDoc, a friendly master mechanic. "
        f"The car's remaining life is {payload.prediction_km:.0f} kilometers. "
        f"The biggest mechanical issue is {payload.worst_sensor}. "
        f"Answer this user question as a helpful mechanic. Be extremely concise, use 1 or 2 short sentences maximum: {payload.question}"
    )

    try:
        response = model.generate_content(context)
        reply = response.text
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=502, detail=f"Chatbot failed to connect: {e}")
    return {"reply": reply}

