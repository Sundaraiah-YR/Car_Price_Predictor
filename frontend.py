import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Car_Price_Prediction", page_icon="🚗")
st.header("Data-Driven Car Price Predictor:")


df = pd.read_csv("Car_Price_Prediction.csv")

with open("Rmodel.pkl", "rb") as file:
    Rmodel = pickle.load(file)

with open("encoders.pkl", "rb") as file:
    encoders = pickle.load(file)

obj_make = encoders["make"]
obj_model = encoders["model"]
obj_fuel = encoders["fuel"]
obj_trans = encoders["trans"]


makes = sorted(df["Make"].unique())
models = sorted(df["Model"].unique())
fuel_types = sorted(df["Fuel Type"].unique())
transmissions = sorted(df["Transmission"].unique())

with st.container(border=True):
    col1, col2 = st.columns(2)

    make = col1.selectbox("Car Brand", options=makes)
    model_name = col2.selectbox("Car Model", options=models)
    year = col1.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1, value=2015)
    engine_size = col2.number_input("Engine Size", min_value=0.5, max_value=6.0, step=0.1, value=1.5)
    mileage = col1.number_input("Mileage (in km)", min_value=0, max_value=500000, step=1000, value=50000)
    fuel_type = col2.selectbox("Fuel Type", options=fuel_types)
    transmission = col1.selectbox("Transmission", options=transmissions)

    input_values = [[
        obj_make.transform([make])[0],
        obj_model.transform([model_name])[0],
        year,
        engine_size,
        mileage,
        obj_fuel.transform([fuel_type])[0],
        obj_trans.transform([transmission])[0]
    ]]

    c1, c2, c3 = st.columns([1.6, 1.5, 1])
    if c2.button("Predict Price"):
        prediction = Rmodel.predict(input_values)[0]
        st.subheader(f"Estimated Price: ₹{prediction}")
