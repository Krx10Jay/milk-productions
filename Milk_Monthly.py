import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
os.makedirs("models", exist_ok=True)

# Streamlit page configuration
st.set_page_config(page_title="Milk Forecast Dashboard", layout="wide")
st.title("Milk Production Forecasting with LSTM")

# Sidebar - Region and settings
st.sidebar.header("⚙️ Settings")
region = st.sidebar.selectbox("Select Region Format", ["Default", "Nigeria (₦/Litre)", "US ($/Gallon)", "EU (€/Litre)"])
forecast_steps = st.sidebar.slider("Months to Forecast", min_value=6, max_value=36, value=12)
lstm1_units = st.sidebar.number_input("LSTM Layer 1 Units", min_value=32, max_value=512, value=150)
use_dropout = st.sidebar.checkbox("Use Dropout?", value=True)

# Format function
def format_unit(value):
    if region == "Nigeria (₦/Litre)":
        return f"₦{value:,.0f} per Litre"
    elif region == "US ($/Gallon)":
        return f"${value:,.2f} per Gallon"
    elif region == "EU (€/Litre)":
        return f"€{value:,.2f} per Litre"
    return f"{value:,.0f}"

# File uploader
uploaded_file = st.file_uploader("Upload CSV with 'Date' and 'Production' columns", type='csv')
if uploaded_file:
    df = pd.read_csv(uploaded_file, index_col='Date', parse_dates=True)
    st.write("Uploaded Data", df.tail())
    data = df['Production'].values.reshape(-1, 1)

    # Define model and scaler paths
    model_path = "models/milk1_lstm_model.h5"
    scaler_path = "models/scaler.save1"
    # scaler_path = "models/scaler.save"
    # model_path = "models/milk_lstm_model.h5"
    # os.makedirs("models", exist_ok=True)

    # Load or create scaler
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        st.success("Loaded saved scaler.")
    else:
        scaler = MinMaxScaler()
        scaler.fit(data)
        joblib.dump(scaler, scaler_path)
        st.info("Scaler fitted and saved.")

    scaled_data = scaler.transform(data)
    n_input = 12
    n_features = 1
    generator = TimeseriesGenerator(scaled_data, scaled_data, length=n_input, batch_size=1)

    # Model training
    model = None
    history = None
    if st.sidebar.button("Train & Save Model"):
        model = Sequential()
        model.add(LSTM(lstm1_units, activation='relu', return_sequences=False, input_shape=(n_input, n_features)))
        if use_dropout:
            model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer=Adam(), loss='mse')

        history = model.fit(generator, epochs=100, verbose=0)
        model.save(model_path)
        st.success("Model trained and saved.")

        st.subheader("Training Loss Curve")
        fig2, ax2 = plt.subplots()
        ax2.plot(history.history['loss'], label='Training Loss (MSE)')
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")
        ax2.set_title("Model Training Loss")
        ax2.legend()
        st.pyplot(fig2)

    elif os.path.exists(model_path):
        if st.sidebar.button("Load Saved Model"):
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
            st.success("Loaded saved model and scaler.")

    # If model is ready
    if model is not None:
        predicted_scaled = model.predict(generator)
        predicted = scaler.inverse_transform(predicted_scaled)
        true = scaler.inverse_transform(scaled_data[n_input:])

        # Error metrics
        mae = mean_absolute_error(true, predicted)
        rmse = np.sqrt(mean_squared_error(true, predicted))
        st.subheader("Error Metrics")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")

        # Historical predictions
        df_pred = df[n_input:].copy()
        df_pred['predicted_production'] = predicted

        # Forecast future
        current_batch = scaled_data[-n_input:].reshape(1, n_input, n_features)
        future_predictions = []
        for _ in range(forecast_steps):
            future_pred = model.predict(current_batch, verbose=0)[0]
            future_predictions.append(future_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[future_pred]], axis=1)

        future_predictions = np.array(future_predictions)
        future_predictions = scaler.inverse_transform(future_predictions)

        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
        future_df = pd.DataFrame({'forecast_production': future_predictions.flatten()}, index=future_dates)

        # Plotting
        st.subheader("Forecast Plot")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df['Production'], label='Actual')
        ax.plot(df_pred.index, df_pred['predicted_production'], label='Predicted')
        ax.plot(future_df.index, future_df['forecast_production'], label='Forecast', linestyle='--')
        ax.set_title("Milk Production Forecast")
        ax.legend()
        st.pyplot(fig)

        # Show forecast sample
        st.subheader("Sample Forecast Values")
        st.write(future_df.head().applymap(lambda x: format_unit(x)))

        # Downloads
        st.subheader("Download Forecast")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download Historical Predictions", df_pred.to_csv().encode(), "historical_predictions.csv", "text/csv")
        with col2:
            st.download_button("Download Future Forecast", future_df.to_csv().encode(), "future_forecast.csv", "text/csv")

