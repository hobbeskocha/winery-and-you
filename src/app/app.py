import streamlit as st
import requests

EMAIL_API_URL = "http://127.0.0.1:8000/predict-email"
NEWSLETTER_API_URL = "http://127.0.0.1:8000/predict-newsletter"
WINEMAKER_API_URL = "http://127.0.0.1:8000/predict-winemaker"


# Header
st.title = "Winery Subscription Predictions"
st.markdown("# Winery Subscription Predictions 🍷")
header_description = """
This simple interface allows you to utilize the predictive models developed from our portfolio project
to predict whether a customer will subscribe to the winery's emails, newsletters, or winemaker calls.\n\n 

Simply input the required information and click the "Predict" button for the respective marketing channel. Behind the scenes,
the most accurate model of the Logistic Regression and Random Forest models is used to make the prediction.
"""

st.write(header_description)
st.markdown("######")

# User Inputs

## Email Subscription
st.markdown("## Email Subscription 📧")
sale_amount = st.number_input("Total Sale Amount", min_value=0.0, max_value=100_000.0, value=5.0, key="email_sale_amount")
order_volume = st.number_input("Order Volume", min_value=0, max_value=100_000, value=5, key="email_order_volume")
customer_segment = st.selectbox("Customer Segment", ["High Roller", "Luxury Estate",
                                                      "Wine Enthusiast", "Casual Visitor"], key="email_customer_segment")
division = st.selectbox("US Division", ["New England", "Middle Atlantic", "East North Central",
                                      "West North Central", "South Atlantic", "East South Central",
                                      "West South Central", "Mountain", "Pacific"], key="email_division")
newsletter_subscr = st.checkbox("Newsletter Subscription", key="email_newsletter_subscr")
winemaker_call_subscr = st.checkbox("Winemaker Call Subscription", key="email_winemaker_subscr")

## Email Predictions
if st.button("Predict Email Subscription"):
    input = {
        "OrderVolume": order_volume,
        "CustomerSegment": customer_segment,
        "Division": division,
        "SaleAmount": sale_amount,
        "NewsletterSubscr": newsletter_subscr,
        "WinemakerCallSubscr": winemaker_call_subscr
    }

    response = requests.post(EMAIL_API_URL, json=input)

    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.success(f"Email Prediction: {prediction}", icon="🎉")
    else:
        st.error("Error: Unable to make Email prediction")

st.markdown("######")


## Newsletter Subscription
st.markdown("## Newsletter Subscription 📰")
sale_amount = st.number_input("Total Sale Amount", min_value=0.0, max_value=100_000.0, value=5.0, key="newsletter_sale_amount")
order_volume = st.number_input("Order Volume", min_value=0, max_value=100_000, value=5, key="newsletter_order_volume")
customer_segment = st.selectbox("Customer Segment", ["High Roller", "Luxury Estate",
                                                      "Wine Enthusiast", "Casual Visitor"], key="newsletter_customer_segment")
division = st.selectbox("US Division", ["New England", "Middle Atlantic", "East North Central",
                                      "West North Central", "South Atlantic", "East South Central",
                                      "West South Central", "Mountain", "Pacific"], key="newsletter_division")
email_subscr = st.checkbox("Email Subscription", key = "newsletter_email_subscr")
winemaker_call_subscr = st.checkbox("Winemaker Call Subscription", key="newsletter_winemaker_subscr")

## Newsletter Predictions
if st.button("Predict Newsletter Subscription"):
    input = {
        "OrderVolume": order_volume,
        "CustomerSegment": customer_segment,
        "Division": division,
        "SaleAmount": sale_amount,
        "WinemakerCallSubscr": winemaker_call_subscr,
        "EmailSubscr": email_subscr
    }

    response = requests.post(NEWSLETTER_API_URL, json=input)

    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.success(f"Newsletter Prediction: {prediction}", icon="🎉")
    else:
        st.error("Error: Unable to make Newsletter prediction")

st.markdown("######")

## Winemaker Call Subscription
st.markdown("## Winemaker Call Subscription 📞")
sale_amount = st.number_input("Total Sale Amount", min_value=0.0, max_value=100_000.0, value=5.0, key="winemaker_sale_amount")
order_volume = st.number_input("Order Volume", min_value=0, max_value=100_000, value=5, key="winemaker_order_volume")
customer_segment = st.selectbox("Customer Segment", ["High Roller", "Luxury Estate",
                                                      "Wine Enthusiast", "Casual Visitor"], key="winemaker_customer_segment")
division = st.selectbox("US Division", ["New England", "Middle Atlantic", "East North Central",
                                      "West North Central", "South Atlantic", "East South Central",
                                      "West South Central", "Mountain", "Pacific"], key="winemaker_division")
email_subscr = st.checkbox("Email Subscription", key="winemaker_email_subscr")
newsletter_subscr = st.checkbox("Newsletter Subscription", key="winemaker_newsletter_subscr")

## Winemaker Predictions
if st.button("Predict Winemaker Subscription"):
    input = {
        "OrderVolume": order_volume,
        "CustomerSegment": customer_segment,
        "Division": division,
        "SaleAmount": sale_amount,
        "NewsletterSubscr": email_subscr,
        "EmailSubscr": winemaker_call_subscr
    }

    response = requests.post(WINEMAKER_API_URL, json=input)

    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.success(f"Winemaker Prediction: {prediction}", icon="🎉")
    else:
        st.error("Error: Unable to make Winemaker prediction")


st.markdown("######")

footer_description = """
### Future updates will include:
1. refining the UX/UI of this interface
2. improving the robustness and accuracy of the underlying models with hyperparameter tuning 
3. introducing multinomial classification to predict a customer's segment e.g. High Roller, Wine Enthusiast, etc.
"""
st.markdown(footer_description)