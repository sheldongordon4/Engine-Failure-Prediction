import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the trained LightGBM model
model = joblib.load('/Users/sheldongordon/Desktop/Engine Failure Prediction/lightgbm_model.pkl')

# Set page configuration
st.set_page_config(page_title="Engine Health Dashboard", layout="wide")

# Title and description
st.title("Engine Health Monitoring Dashboard")
st.markdown("Monitor, assess, and predict the health of turbofan engines using NASA CMAPSS data.")

# Sidebar inputs
st.sidebar.header("Configuration")
threshold = st.sidebar.slider("RUL Threshold (for Inspection)", 10, 50, 30)
uploaded_file = st.sidebar.file_uploader("Upload Engine Sensor CSV", type=["csv"])

# Main logic
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Extract and save engine IDs
    engine_ids = df['engine_id'].unique()

    # Load feature names used during training
    features_to_drop = ['engine_id', 'op_set_1', 'op_set_2', 'op_set_3', 'T2', 'P2', 'P15', 'epr', 'farB', 'Nf_dmd', 'PCNfR_dmd']
    df_predict = df.copy()
    df_predict = df.drop(columns=features_to_drop)
    
    features = df_predict.columns.to_list()
    
    sensor_columns = [col for col in features if col != 'cycle_time']
    selected_sensor = st.sidebar.selectbox("Sensor to Monitor", sensor_columns)
    
    # Predict RUL
    df_predict['predicted_rul'] = model.predict(df_predict[features])

    # Categorize engine health status
    df_predict['action'] = df_predict['predicted_rul'].apply(
        lambda x: 'Repair' if x <= 15 else ('Inspect' if x <= threshold else 'Monitor')
    )

    # Metrics overview
    st.metric("Average Predicted RUL", round(df_predict["predicted_rul"].mean(), 1))
    st.metric("Engines Needing Repair", int((df_predict["action"] == "Repair").sum()))

    # Highlight function for DataFrame
    def highlight_action(val):
        colors = {"Repair": "#FFCCCC", "Inspect": "#FFF5CC", "Monitor": "#CCFFCC"}
        return f'background-color: {colors.get(val, "")}'


    # Health table
    st.header("Engine Health Overview")
    st.dataframe(
        df_predict.sort_values("predicted_rul").style.applymap(highlight_action, subset=["action"]),
        use_container_width=True
    )

    # Histogram of risk
    st.subheader("Risk Assessment")
    fig1, ax1 = plt.subplots()
    sns.histplot(data=df_predict, x="predicted_rul", hue="action", multiple="stack", palette="muted", ax=ax1)
    st.pyplot(fig1)

    # Engine-specific diagnostics
    selected_engine = st.selectbox("Drill-down: Select Engine ID", engine_ids)

    # Filter and plot selected engine's sensor trend
    engine_df = df[df.engine_id == selected_engine].sort_values("cycle_time")
    if selected_sensor in sensor_columns:
        sensor_data = engine_df[["cycle_time", selected_sensor]]

        st.subheader(f"{selected_sensor} Over Time for Engine {selected_engine}")
        fig2, ax2 = plt.subplots()
        sns.lineplot(data=sensor_data, x="cycle_time", y=selected_sensor, ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning(f"{selected_sensor} not found for Engine {selected_engine}")

    # Maintenance recommendation
    latest_rul = df_predict[df['engine_id'] == selected_engine].sort_values("cycle_time")["predicted_rul"].iloc[-1]
    if latest_rul <= 15:
        st.error(f"⚠️ Engine {selected_engine} should be repaired immediately! RUL: {latest_rul}")
    elif latest_rul <= threshold:
        st.warning(f"Engine {selected_engine} needs inspection soon. RUL: {latest_rul}")
    else:
        st.success(f"Engine {selected_engine} is operating normally. RUL: {latest_rul}")

else:
    st.info("Please upload a CSV file to begin analysis.")
