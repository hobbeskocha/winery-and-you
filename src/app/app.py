import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.title = "Winery Subscription Predictions"
st.write("This is a simple app to predict whether a customer will subscribe to the winery email.")

# User Inputs
order_volume = st.number_input("Order Volume", min_value=0, max_value=100_000, value=5)
customer_segment = st.selectbox("Customer Segment", ["High Roller", "Luxury Estate",
                                                      "Wine Enthusiast", "Casual Visitor"])
division = st.selectbox("Division", ["New England", "Middle Atlantic", "East North Central",
                                      "West North Central", "South Atlantic", "East South Central",
                                      "West South Central", "Mountain", "Pacific"])
sale_amount = st.number_input("Sale Amount", min_value=0.0, max_value=100_000.0, value=5)
newsletter_subscr = st.checkbox("Newsletter Subscription")
winemaker_call_subscr = st.checkbox("Winemaker Call Subscription")

# Make predictions
if st.button("Predict"):
    input = {
        "OrderVolume": order_volume,
        "CustomerSegment": customer_segment,
        "Division": division,
        "SaleAmount": sale_amount,
        "NewsletterSubscr": newsletter_subscr,
        "WinemakerCallSubscr": winemaker_call_subscr
    }

    response = requests.post(API_URL, json=input)

    if response.status_code == 200:
        prediction = response.json(["prediction"])
        st.success(f"Prediction: {prediction}")
    else:
        st.error("Error: Unable to make prediction")

