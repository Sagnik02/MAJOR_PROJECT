import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Title and description
st.title("Charge Capacity Prediction App")
st.write("Predict the Charge Capacity (Ah) of a battery based on various parameters.")

# Load dataset
@st.cache
def load_dataset():
    # Replace this path with the location of your dataset
    file_path = "data.xls" 
    data = pd.read_excel(file_path)
    return data

# Load and display data
df = load_dataset()
st.write("Dataset Preview:")
st.dataframe(df)

# Split data into features and target
X = df.drop(columns=["Charge_Capacity(Ah)"])  # Features
y = df["Charge_Capacity(Ah)"]  # Target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Train model
model = SVR(kernel='rbf')
model.fit(X_train, y_train)

# Test model performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("### Model Performance on Test Data")
st.write(f"Mean Squared Error: {mse:.4f}")
st.write(f"R² Score: {r2:.4f}")

# Prediction inputs
st.write("### Input Parameters for Prediction")
current = st.number_input("Current (A)", min_value=-3.0, step=0.01, value=1.0)
voltage = st.number_input("Voltage (V)", min_value=0.0, step=0.1, value=3.5)
charge_energy = st.number_input("Charge Energy (Wh)", min_value=0.0, step=0.001, value=5.0)
discharge_energy = st.number_input("Discharge Energy (Wh)", min_value=0.0, step=0.001, value=5.0)
dv_dt = st.number_input("dV/dt (V/s)", min_value=-1.0, step=0.001, value=0.0)
# Make prediction
if st.button("Predict Charge Capacity"):
    input_features = np.array([[current, voltage, charge_energy, discharge_energy, dv_dt]])
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Charge Capacity (Ah): {prediction[0]:.6f}")
