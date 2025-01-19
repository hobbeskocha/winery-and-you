import streamlit as st
import requests

# # Local APIs
# EMAIL_API_URL = "http://127.0.0.1:8080/predict-email"
# NEWSLETTER_API_URL = "http://127.0.0.1:8080/predict-newsletter"
# WINEMAKER_API_URL = "http://127.0.0.1:8080/predict-winemaker"

# Cloud APIs
GOOGLE_CLOUD_API_URL = "https://winery-be-505678819204.us-east1.run.app"
EMAIL_API_URL = GOOGLE_CLOUD_API_URL + "/predict-email"
NEWSLETTER_API_URL = GOOGLE_CLOUD_API_URL + "/predict-newsletter"
WINEMAKER_API_URL = GOOGLE_CLOUD_API_URL + "/predict-winemaker"

# Header
st.title = "Winery Subscription Predictions"
st.markdown("# A Winery's Marketing Edge: Predict Subscriptions with AI 🍷")
header_description = """
This interface leverages machine learning models to predict whether a customer will subscribe 
to one of the winery's marketing channels - emails, newsletters, or winemaker calls - 
based on their subscription status to the other 2 channels, total spending, order volume, customer segment, and geographic location.\n\n 
Simply enter the customer's details and click "Predict" to find out if the customer will subscribe to that channel. 
Our system uses the most accurate model, between Logistic Regression and Random Forest, to make its prediction.
"""

st.write(header_description)
st.markdown("######")

# User Inputs

## Email Subscription
st.markdown("## Email Subscription 📧")
sale_amount = st.number_input("Total Spending (in USD)", min_value=0.0, max_value=50_000.0, value=250.0, key="email_sale_amount")
order_volume = st.number_input("Number of Orders", min_value=0, max_value=25, value=3, key="email_order_volume")
customer_segment = st.selectbox("Customer Segment", ["High Roller", "Luxury Estate",
                                                      "Wine Enthusiast", "Casual Visitor"], key="email_customer_segment")
division = st.selectbox("US Division", ["New England", "Middle Atlantic", "East North Central",
                                      "West North Central", "South Atlantic", "East South Central",
                                      "West South Central", "Mountain", "Pacific"], key="email_division")
newsletter_subscr = st.checkbox("Has Newsletter Subscription", key="email_newsletter_subscr")
winemaker_call_subscr = st.checkbox("Has Winemaker Call Subscription", key="email_winemaker_subscr")

## Email Predictions
if st.button("Predict", key = "email_predict"):
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
        if prediction == True:
            st.success(f"Yes, will subscribe to Emails!", icon="🎉")
        else:
            st.success(f"No, will not subscribe to Emails.", icon="😞")
    else:
        st.error("Error: Unable to make Email prediction")

st.markdown("######")


## Newsletter Subscription
st.markdown("## Newsletter Subscription 📰")
sale_amount = st.number_input("Total Spending (in USD)", min_value=0.0, max_value=50_000.0, value=250.0, key="newsletter_sale_amount")
order_volume = st.number_input("Number of Orders", min_value=0, max_value=25, value=3, key="newsletter_order_volume")
customer_segment = st.selectbox("Customer Segment", ["High Roller", "Luxury Estate",
                                                      "Wine Enthusiast", "Casual Visitor"], key="newsletter_customer_segment")
division = st.selectbox("US Division", ["New England", "Middle Atlantic", "East North Central",
                                      "West North Central", "South Atlantic", "East South Central",
                                      "West South Central", "Mountain", "Pacific"], key="newsletter_division")
email_subscr = st.checkbox("Has Email Subscription", key = "newsletter_email_subscr")
winemaker_call_subscr = st.checkbox("Has Winemaker Call Subscription", key="newsletter_winemaker_subscr")

## Newsletter Predictions
if st.button("Predict", key="newsletter_predict"):
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
        if prediction == True:
            st.success(f"Yes, will subscribe to Newsletters!", icon="🎉")
        else:
            st.success(f"No, will not subscribe to Newsletters.", icon="😞")
    else:
        st.error("Error: Unable to make Newsletter prediction")

st.markdown("######")

## Winemaker Call Subscription
st.markdown("## Winemaker Call Subscription 📞")
sale_amount = st.number_input("Total Spending (in USD)", min_value=0.0, max_value=50_000.0, value=250.0, key="winemaker_sale_amount")
order_volume = st.number_input("Number of Orders", min_value=0, max_value=25, value=3, key="winemaker_order_volume")
customer_segment = st.selectbox("Customer Segment", ["High Roller", "Luxury Estate",
                                                      "Wine Enthusiast", "Casual Visitor"], key="winemaker_customer_segment")
division = st.selectbox("US Division", ["New England", "Middle Atlantic", "East North Central",
                                      "West North Central", "South Atlantic", "East South Central",
                                      "West South Central", "Mountain", "Pacific"], key="winemaker_division")
email_subscr = st.checkbox("Has Email Subscription", key="winemaker_email_subscr")
newsletter_subscr = st.checkbox("Has Newsletter Subscription", key="winemaker_newsletter_subscr")

## Winemaker Predictions
if st.button("Predict", key="winemaker_predict"):
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
        if prediction == True:
            st.success(f"Yes, will subscribe to Winemaker Calls!", icon="🎉")
        else:
            st.success(f"No, will not subscribe to Winemaker Calls.", icon="😞")
    else:
        st.error("Error: Unable to make Winemaker prediction")


st.markdown("######")

footer_description = """
### Future updates will include:
1. Refining the UX/UI of this interface
2. Improving the robustness and accuracy of the underlying models with hyperparameter tuning 
3. Introducing multinomial classification to predict a customer's segment e.g. High Roller, Wine Enthusiast, etc.
"""
st.markdown(footer_description)