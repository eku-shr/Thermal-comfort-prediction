import streamlit as st
import numpy as np
import joblib

# Load your trained model
model = joblib.load("best_gradient_boosting_model.pkl")

# Clothing options with CLO values (add more from your chart if needed)
clo_options = {
    "Half Cotton T-Shirt": 0.36,
    "Full Cotton Pant": 0.24,
    "Cotton Kurta": 0.36,
    "Salwar": 0.28,
    "Shirt": 0.61,
    "Maxi": 0.45,
    "Winter Jacket": 0.96,
    "Upper Tracksuit": 0.34,
    "Tracksuit Trouser": 0.28,
    "Polyester Shorts": 0.06,
    "Woolen Highnecks": 0.34,
    "Thick Trousers": 0.28,
    "Woolen Sweaters": 0.34,
    "Woolen Caps": 0.10,
    "Sando": 0.13,
    "Dhoti": 0.28,
    "Heavy Winter Jackets": 1.00,
    "Light Winter Jackets": 0.96,
    "Suti Sari": 0.67,
    "Furred Trousers": 0.28,
    "Thick Jackets": 0.96,
    "Permeable Cotton Shirt": 0.61,
    "Jumper": 0.34,
    "Shawl": 0.20,
    "Thin Sweaters": 0.20,
    "Thermocoat Upper": 1.00,
    "Thermocoat Lowers": 0.28,
    "Gloves": 0.00,
    "Thin Woolen Uppers": 0.20,
    "Thin Woolen Lowers": 0.28,
    "Thick Woolen Uppers": 0.34,
    "Thick Woolen Lowers": 0.28,
    "Woolen Hoodie": 0.34,
    "Thin Sleeveless Coat": 0.20,
    "Kamij": 0.61,
    "Woolen Cholo": 0.34,
    "Fariya": 0.30,
    "Patuki": 0.02,
    "Pachhyaura": 0.20,
    "Men's underwear": 0.05,
    "Women's underwear": 0.03,
    "Men's Upper Innerwear": 0.08,
    "Women's Upper Innerwear": 0.20,
    "Blouses": 0.27,
    "Socks": 0.05
}


# Activity options with MET values
met_options = {
   "Sleeping": 0.7,
    "Reclining": 0.8,
    "Seated, quiet": 1.0,
    "Standing, relaxed": 1.2,
    "Walking (3.2 km/h)": 2.0,
    "Walking (4.3 km/h)": 2.6,
    "Walking (6.4 km/h)": 3.8,
    "Reading, seated": 1.0,
    "Writing": 1.0,
    "Typing": 1.1,
    "Filing, seated": 1.2,
    "Filing, standing": 1.4,
    "Walking about (office)": 1.7,
    "Lifting/packing": 2.1,
    "Driving car": 1.5,
    "Driving aircraft (routine)": 1.2,
    "Aircraft, instrument landing": 1.8,
    "Aircraft, combat": 2.4,
    "Heavy vehicle": 3.2,
    "Cooking": 1.8,
    "Housecleaning": 3.0,
    "Seated, heavy limb movement": 2.2,
    "Machine work (sawing)": 1.8,
    "Machine work (light)": 2.2,
    "Machine work (heavy)": 4.0,
    "Handling 50 kg bags": 4.0,
    "Pick and shovel work": 4.4,
    "Dancing, social": 3.4,
    "Calisthenics/exercise": 3.5,
    "Tennis, singles": 4.0,
    "Basketball": 6.3,
    "Wrestling, competitive": 8.0
}


# Streamlit app layout
st.set_page_config(page_title="Thermal Comfort Predictor", layout="centered")
st.title("🌡️ Thermal Comfort Predictor")
st.markdown("This app predicts the thermal comfort using **Predicted Mean Vote (PMV)** based on environment, clothing, and activity.")

# --- Environmental Inputs ---
st.header("Environmental Inputs")
temperature = st.number_input("Temperature (°C)", min_value=-15.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)

# --- Personal Factors ---
st.header("Personal Inputs")

# Multi-select for clothing
selected_clothes = st.multiselect("Select Clothing Items Worn", list(clo_options.keys()))
clo_value = sum([clo_options[item] for item in selected_clothes])

# Dropdown for activity
selected_met = st.selectbox("Select Activity Type", list(met_options.keys()))
met_value = met_options[selected_met]

# Predict button
if st.button("Predict PMV"):
    input_data = np.array([[temperature, humidity, met_value, clo_value]])
    predicted_pmv = model.predict(input_data)[0]

    st.subheader("🧮 Prediction Results")
    st.success(f"**Predicted PMV:** {predicted_pmv:.2f}")
    st.info(f"**Total CLO value:** {clo_value:.2f} | **Selected MET value:** {met_value}")

    # ASHRAE Thermal Sensation Scale
    if predicted_pmv >= 3:
        sensation = "Hot"
        color = "🔥"
    elif 2 <= predicted_pmv < 3:
        sensation = "Warm"
        color = "🌡️"
    elif 1 <= predicted_pmv < 2:
        sensation = "Slightly Warm"
        color = "🟥"
    elif -1 < predicted_pmv < 1:
        sensation = "Neutral"
        color = "🟦"
    elif -2 < predicted_pmv <= -1:
        sensation = "Slightly Cool"
        color = "🟦"
    elif -3 < predicted_pmv <= -2:
        sensation = "Cool"
        color = "❄️"
    else:  # PMV <= -3
        sensation = "Cold"
        color = "🧊"

    st.markdown(f"**Thermal Sensation (ASHRAE scale):** {color} {sensation}")
