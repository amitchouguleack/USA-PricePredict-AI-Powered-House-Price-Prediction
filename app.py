import streamlit as st
import pandas as pd
import joblib  # if you're loading a saved model
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("USA_Housing.csv")

# Title
st.title("🏡 USA House Price Estimator")

# Sidebar inputs
income = st.slider("Average Area Income", 20000, 150000, 50000)
house_age = st.slider("Average House Age", 1, 50, 10)
rooms = st.slider("Average Number of Rooms", 1, 10, 5)
bedrooms = st.slider("Average Number of Bedrooms", 1, 5, 3)
population = st.slider("Area Population", 1000, 100000, 30000)

# Prediction
model = RandomForestRegressor()
model.fit(df.drop(["Price", "Address"], axis=1), df["Price"])
input_data = [[income, house_age, rooms, bedrooms, population]]
prediction = model.predict(input_data)[0]

# Output
st.subheader(f"Estimated House Price: ${prediction:,.2f}")

st.markdown("---")
st.markdown("Made with ❤️ by Amit Chougule")

st.markdown("---")
st.markdown("🔗 [View Source on GitHub](https://github.com/amitchouguleack)")
st.markdown("Made with ❤️ by Amit Chougule")
