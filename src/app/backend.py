from fastapi import FastAPI
import joblib
import logging

from classes.EmailInputData import EmailInputData
from classes.NewsletterInputData import NewsletterInputData
from classes.WinemakerInputData import WinemakerInputData
from helpers.categorical_encoders import *

# Load models from local storage
model_log_email = joblib.load("../model-artifacts/log_email.pkl")
model_log_newsletter = joblib.load("../model-artifacts/log_newsletter.pkl")
model_log_winemaker = joblib.load("../model-artifacts/log_winemaker.pkl")

# Create backend app
backend_app = FastAPI()
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


# HTTP requests
@backend_app.post("/predict-email")
def predict_email(data: EmailInputData):
    # map categorical data for one-hot encoding
    is_high_roller, is_luxury_estate, is_wine_enthusiast = encode_customer_segment(data.CustomerSegment)
    (is_east_south_central, is_middle_atlantic, is_mountain,
      is_new_england, is_pacific, is_south_atlantic,
        is_west_north_central, is_west_south_central) = encode_division(data.Division)

    # config input data for model prediction
    input_data = [[1.0, data.OrderVolume, data.SaleAmount, 
                   is_high_roller, is_luxury_estate, is_wine_enthusiast, 
                   is_east_south_central, is_middle_atlantic, is_mountain,
                   is_new_england, is_pacific, is_south_atlantic,
                    is_west_north_central, is_west_south_central,
                    data.NewsletterSubscr, data.WinemakerCallSubscr]]
    
    prediction = model_log_email.predict(input_data)
    logging.info(f"Email Probability: {prediction} and Prediction: {round(prediction[0])}")
    return {"prediction": round(prediction[0])}


@backend_app.post("/predict-newsletter")
def predict_newsletter(data: NewsletterInputData):
    # map categorical data for one-hot encoding
    is_high_roller, is_luxury_estate, is_wine_enthusiast = encode_customer_segment(data.CustomerSegment)
    (is_east_south_central, is_middle_atlantic, is_mountain,
      is_new_england, is_pacific, is_south_atlantic,
        is_west_north_central, is_west_south_central) = encode_division(data.Division)
    
    # config input data for model prediction
    input_data = [[1.0, data.OrderVolume, data.SaleAmount, 
                   is_high_roller, is_luxury_estate, is_wine_enthusiast, 
                   is_east_south_central, is_middle_atlantic, is_mountain,
                   is_new_england, is_pacific, is_south_atlantic,
                    is_west_north_central, is_west_south_central,
                    data.WinemakerCallSubscr, data.EmailSubscr]]
    
    prediction = model_log_newsletter.predict(input_data)
    logging.info(f"Newsletter Probability: {prediction} and Prediction: {round(prediction[0])}")
    return {"prediction": round(prediction[0])}


@backend_app.post("/predict-winemaker")
def predict_winemaker(data: WinemakerInputData):
    # map categorical data for one-hot encoding
    is_high_roller, is_luxury_estate, is_wine_enthusiast = encode_customer_segment(data.CustomerSegment)
    (is_east_south_central, is_middle_atlantic, is_mountain,
      is_new_england, is_pacific, is_south_atlantic,
        is_west_north_central, is_west_south_central) = encode_division(data.Division)
    
    # config input data for model prediction
    input_data = [[1.0, data.OrderVolume, data.SaleAmount, 
                   is_high_roller, is_luxury_estate, is_wine_enthusiast, 
                   is_east_south_central, is_middle_atlantic, is_mountain,
                   is_new_england, is_pacific, is_south_atlantic,
                    is_west_north_central, is_west_south_central,
                    data.NewsletterSubscr, data.EmailSubscr]]
    
    prediction = model_log_winemaker.predict(input_data)
    logging.info(f"Winemaker Probability: {prediction} and Prediction: {round(prediction[0])}")
    return {"prediction": round(prediction[0])}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(backend_app, host="0.0.0.0", port=8000)