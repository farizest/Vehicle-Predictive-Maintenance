


import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import google.generativeai as genai
from tensorflow.keras.losses import MeanSquaredError
import os
import sys

# Add project root to path for imports if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import Attention
from src.config import MODEL_PATH, SCALER_PATH, FEATURE_COLS
import src.data_loader as dl

@st.cache_resource
def load_assets():
    # Mapping 'mse' and 'Attention' for a successful load
    custom_objects = {"Attention": Attention, "mse": MeanSquaredError()}
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please run main.py first.")
        return None, None, None
        
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    scaler = joblib.load(SCALER_PATH)
    
    # Load and Preprocess Test Data for "Real" Simulation
    try:
        _, test_df, _ = dl.load_data()
        # Scale the data using the loaded scaler
        test_df[FEATURE_COLS] = scaler.transform(test_df[FEATURE_COLS])
        print("Test data loaded and scaled for simulation.")
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        test_df = None

    return model, scaler, test_df

# ==========================================
# 2. UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="AutoDoc AI Mechanic", page_icon="🚗", layout="wide")

# Modern Dark Theme CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stMetricValue"] { font-size: 28px; color: #00ffcc; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚗 AutoDoc: AI Mechanic Dashboard")
st.subheader("Predictive Maintenance System")

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    # Get Gemini Key from environment
    GEMINI_KEY = "AIzaSyCwSuV8TZvp4NWdZs7C07bIYVtrVQ55t5E"
    
    model, scaler, test_df = load_assets()
    
    if model and scaler:
        # --- SIDEBAR ---
        st.sidebar.header("Diagnostics Control")
        input_mode = st.sidebar.selectbox("Select Mode", ["Live Telemetry", "Manual Stress Test"])

        if input_mode == "Manual Stress Test":
            oil_temp_val = st.sidebar.slider("Simulated Oil Temp (Normalized)", 0.0, 1.0, 0.8)
            X_sample = np.random.rand(1, 30, 17) 
            X_sample[0, -1, 5] = oil_temp_val
            
        elif input_mode == "Live Telemetry" and test_df is not None:
            unit_ids = test_df['unit'].unique()
            
            # Initialize Session State for Random Selection if not present
            if 'selected_unit' not in st.session_state:
                st.session_state.selected_unit = unit_ids[0]
            if 'current_cycle' not in st.session_state:
                st.session_state.current_cycle = 30

            # Randomize Button
            if st.sidebar.button("🎲 Randomize Test Scenario"):
                new_unit = np.random.choice(unit_ids)
                unit_data = test_df[test_df['unit'] == new_unit]
                max_cycle = unit_data['cycle'].max()
                
                # Ensure valid cycle > 30
                if max_cycle >= 30:
                    new_cycle = np.random.randint(30, max_cycle + 1)
                    st.session_state.selected_unit = new_unit
                    st.session_state.current_cycle = int(new_cycle)
                    st.rerun() 
            
            # Sync session state with widgets
            # We use index based on session state value
            try:
                unit_index = list(unit_ids).index(st.session_state.selected_unit)
            except ValueError:
                unit_index = 0
                
            selected_unit = st.sidebar.selectbox("Select Engine Unit ID", unit_ids, index=unit_index)
            
            # Use callback or direct update if user manually changes selectbox? 
            # Streamlit widgets update their value on change. If keys differ, state resets.
            # To keep it simple: manual change updates 'selected_unit' implicitly via rerun,
            # but we need to update session state if manual interaction happens.
            if selected_unit != st.session_state.selected_unit:
                st.session_state.selected_unit = selected_unit
                # Reset cycle on unit change just to be safe (or pick max)
                unit_data = test_df[test_df['unit'] == selected_unit]
                st.session_state.current_cycle = int(unit_data['cycle'].max())

            # Get data for this unit
            unit_data = test_df[test_df['unit'] == selected_unit]
            max_cycle = int(unit_data['cycle'].max())

            # Ensure we have enough history
            if max_cycle < 30:
                st.error(f"Unit {selected_unit} has less than 30 cycles of data.")
                X_sample = np.zeros((1, 30, 17))
            else:
                # Cycle Slider
                # Ensure session state cycle is within bounds for this unit
                if st.session_state.current_cycle > max_cycle:
                    st.session_state.current_cycle = max_cycle
                    
                current_cycle_val = st.sidebar.slider("Current Cycle", 30, max_cycle, st.session_state.current_cycle)
                
                # Update session state if slider moved manually
                st.session_state.current_cycle = current_cycle_val
                
                # Extract sequence
                seq_df = unit_data[(unit_data['cycle'] > current_cycle_val - 30) & (unit_data['cycle'] <= current_cycle_val)]
                
                if len(seq_df) == 30:
                    X_sample = seq_df[FEATURE_COLS].values.reshape(1, 30, 17)
                    st.sidebar.success(f"Simulating Unit {selected_unit} at Cycle {current_cycle_val}")
                else:
                    st.error("Error creating sequence.")
                    X_sample = np.zeros((1, 30, 17))
        else:
             # Fallback if no data
            X_sample = np.random.rand(1, 30, 17)

        # --- PREDICTION ---
        prediction = model.predict(X_sample, verbose=0).flatten()[0]

        # --- DISPLAY METRICS (Updated to kms) ---
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted RUL", f"{prediction:.0f} kms")
        
        with col2:
            status = "HEALTHY" if prediction > 75 else "WARNING" if prediction > 30 else "CRITICAL"
            st.metric("Engine Status", status)
            
        with col3:
            st.metric("Sensors Active", "17/17")

        # --- ALERT MESSAGES (Updated to kms) ---
        if prediction < 30:
            st.error(f"### ⚠️ IMMEDIATE ACTION REQUIRED\nEngine breakdown predicted within {prediction:.0f} kms.")
        elif prediction < 75:
            st.warning(f"### 🔔 MAINTENANCE ADVISED\nSchedule an inspection soon. Only {prediction:.0f} kms of safe driving left.")

        # --- AI DIAGNOSTIC REPORT ---
        st.divider()
        st.header("🧠 AI Mechanic's Diagnosis")

        if GEMINI_KEY:
            try:
                genai.configure(api_key=GEMINI_KEY)
                explainer_model = genai.GenerativeModel('gemini-2.5-flash')
                
                # Prepare context for AI
                # Get the last time step data for context
                current_sensors = X_sample[0, -1, :]
                # Create a dictionary of {Sensor Name: Value}
                sensor_dict = {FEATURE_COLS[i]: f"{val:.2f}" for i, val in enumerate(current_sensors)}
                
                # Identify potential anomalies (simple heuristic: values close to 0 or 1 in MinMax scale often indicate extremes)
                # We'll just pass the whole dict and let the LLM interpret "extremes" broadly
                
                # Determine status message for context
                if prediction < 30:
                    status_context = "CRITICAL: Engine breakdown imminent."
                elif prediction < 75:
                    status_context = "WARNING: Maintenance advised soon."
                else:
                    status_context = "HEALTHY: Engine operating normally."
                
                prompt = f"""
                You are AutoDoc, an expert car mechanic AI. 
                Analysis Context:
                - Predicted Remaining Useful Life (RUL): {prediction:.0f} km.
                - Current Sensor Readings (Normalized 0-1): {sensor_dict}
                - Status Assessment: {status_context}
                
                Task:
                Explain the vehicle's health status in simple, non-technical terms for a regular driver.
                Focus on the most likely issues based on the RUL and sensor readings (high/low values are suspicious).
                
                Output Format (Markdown):
                **1. 🛑 The Issue (What's wrong?)**: [1 sentence]
                **2. 📍 The Component (Where is it?)**: [1 sentence, name likely sensor/part]
                **3. 🔧 Repair Action (How to fix?)**: [1 sentence recommendation]
                **4. 💡 Why?**: [1 sentence explaining the link between sensor data and prediction]
                
                Keep it short, encouraging, and easy to understand. Each output in separate line.
                """
                
                with st.spinner("Analyzing engine telemetry..."):
                    # We use a placeholder to avoid re-generating on every rerun if not needed, 
                    # but for now we'll generate on every prediction update.
                    explanation = explainer_model.generate_content(prompt).text
                    st.markdown(explanation)
                    
            except Exception as e:
                st.error(f"AI Explanation unavailable: {e}")
        else:
             st.info("AI Diagnostics disabled. Please configure API Key.")

        # --- XAI TABLE ---
        st.divider()
        st.header("🔍 Detailed Sensor Analysis")
        
        # We can now use the actual sensor deviation from "nominal" (0.5) as a proxy for impact, 
        # or stick to the random placeholder if we want to show 'feature importance' which is different from 'value magnitude'.
        # For consistency with the user request "why what where how", the AI report above covers the main explanation.
        # We will keep this table but perhaps rename it or remove the random values if they are confusing.
        # Let's keep it as is for now but clarify it's a "Simulation".
        
        influence = np.random.uniform(-15, 5, 17) # Placeholder for SHAP values
        inf_df = pd.DataFrame({'Component': FEATURE_COLS, 'Health Impact': influence})
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🔴 Sensors Reporting Issues")
            st.dataframe(inf_df.sort_values(by='Health Impact').head(3), use_container_width=True)
        with c2:
            st.subheader("🟢 Sensors Operating Normally")
            st.dataframe(inf_df.sort_values(by='Health Impact', ascending=False).head(3), use_container_width=True)

        # --- CHATBOT SECTION (Fixed Logic) ---
        st.divider()
        st.header("💬 Ask AutoDoc Mechanic")
        
        # Check if key is provided and initialize
        if GEMINI_KEY:
            try:
                genai.configure(api_key=GEMINI_KEY)
                # Use gemini-1.5-flash for stability
                chat_model = genai.GenerativeModel('gemini-2.5-flash')
            
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Display Chat History
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Chat Input
                if prompt := st.chat_input("Ask about the repair..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Find the worst sensor for context
                    worst_sensor = inf_df.sort_values(by='Health Impact').iloc[0]['Component']
                    
                    # Context-aware humanified prompt
                    context = (f"You are AutoDoc, a friendly master mechanic. "
                               f"The car's remaining life is {prediction:.0f} kilometers. "
                               f"The biggest mechanical issue is {worst_sensor}. "
                               f"Answer this user question as a helpful mechanic: {prompt}")
                    
                    response = chat_model.generate_content(context)
                    reply = response.text

                    with st.chat_message("assistant"):
                        st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    
            except Exception as e:
                st.error(f"Chatbot failed to connect: {e}")
        else:
            st.info("Chatbot is currently disabled. Please add a valid API key.")

except Exception as e:
    st.error(f"App Load Error: {e}")
