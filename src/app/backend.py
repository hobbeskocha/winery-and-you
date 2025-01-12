from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import logging

model_log_email = joblib.load("../model-artifacts/log_email.pkl")
model_log_newsletter = joblib.load("../model-artifacts/log_newsletter.pkl")
model_log_winemaker = joblib.load("../model-artifacts/log_winemaker.pkl")

backend_app = FastAPI()
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

class InputData(BaseModel):
    OrderVolume: int
    CustomerSegment: str 
    Division: str 
    SaleAmount: float

class EmailInputData(InputData):
    NewsletterSubscr: bool 
    WinemakerCallSubscr: bool 

class WinemakerInputData(InputData):
    NewsletterSubscr: bool
    EmailSubscr: bool

class NewsletterInputData(InputData):
    WinemakerCallSubscr: bool 
    EmailSubscr: bool

def encode_customer_segment(segment: str):
    is_high_roller = 1 if segment == "High Roller" else 0
    is_luxury_estate = 1 if segment == "Luxury Estate" else 0
    is_wine_enthusiast = 1 if segment == "Wine Enthusiast" else 0
    return [is_high_roller, is_luxury_estate, is_wine_enthusiast]

def encode_division(division: str):
    is_east_south_central = 1 if division == "East South Central" else 0
    is_middle_atlantic = 1 if division == "Middle Atlantic" else 0
    is_mountain = 1 if division == "Mountain" else 0
    is_new_england = 1 if division == "New England" else 0
    is_pacific = 1 if division == "Pacific" else 0
    is_south_atlantic = 1 if division == "South Atlantic" else 0
    is_west_north_central = 1 if division == "West North Central" else 0
    is_west_south_central = 1 if division == "West South Central" else 0

    return [is_east_south_central, is_middle_atlantic, is_mountain,
             is_new_england, is_pacific, is_south_atlantic,
               is_west_north_central, is_west_south_central]


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