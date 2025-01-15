from fastapi import FastAPI
import joblib
import logging
from typing import List

from classes.EmailInputData import EmailInputData
from classes.NewsletterInputData import NewsletterInputData
from classes.WinemakerInputData import WinemakerInputData
from helpers.categorical_encoders import *

# Load models from local storage
model_log_email = joblib.load("../model-artifacts/log_email.pkl")
model_log_newsletter = joblib.load("../model-artifacts/log_newsletter.pkl")
model_log_winemaker = joblib.load("../model-artifacts/log_winemaker.pkl")
model_rf_email = joblib.load("../model-artifacts/rf_email.pkl")
model_rf_newsletter = joblib.load("../model-artifacts/rf_newsletter.pkl")
model_rf_winemaker = joblib.load("../model-artifacts/rf_winemaker.pkl")

def model_selector():
	"""
	compare model accuracies and return best model and model type as tuple
	"""
	metrics_file = open("../model-artifacts/model-metrics.txt")    
	metrics_line = metrics_file.readline()
	metrics_file.close()

	models = metrics_line.split(",")
	email_models = [model for model in models if "Email" in model]
	newsletter_models = [model for model in models if "Newsletter" in model]
	winemaker_models = [model for model in models if "Winemaker" in model]

	best_email_model = [model.split(":")[1] for model in email_models]
	best_email_model = (model_log_email, "Logit") if best_email_model[0] > best_email_model[1] else (model_rf_email, "RF")

	best_newsletter_model = [model.split(":")[1] for model in newsletter_models]
	best_newsletter_model = (model_log_newsletter, "Logit") if best_newsletter_model[0] > best_newsletter_model[1] else (model_rf_newsletter, "RF")

	best_winemaker_model = [model.split(":")[1] for model in winemaker_models]
	best_winemaker_model = (model_log_winemaker, "Logit") if best_winemaker_model[0] > best_winemaker_model[1] else (model_rf_winemaker, "RF")
	
	# print(best_email_model, best_newsletter_model, best_winemaker_model)
	return best_email_model, best_newsletter_model, best_winemaker_model

# Select best models
email_model_tuple, newsletter_model_tuple, winemaker_model_tuple = model_selector()

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
	input_data = [[data.OrderVolume, data.SaleAmount, 
				   is_high_roller, is_luxury_estate, is_wine_enthusiast, 
				   is_east_south_central, is_middle_atlantic, is_mountain,
				   is_new_england, is_pacific, is_south_atlantic,
					is_west_north_central, is_west_south_central,
					data.NewsletterSubscr, data.WinemakerCallSubscr]]
	
	email_model, email_model_type = email_model_tuple
	if email_model_type == "Logit":
		input_data[0].insert(0, 1.0)
	
	prediction = email_model.predict(input_data)
	if email_model_type == "RF":
		logging.info(f"RF Email Prediction: {prediction}")
		return {"prediction": int(prediction[0])}
	else:    
		logging.info(f"Logit Email Probability: {prediction} and Prediction: {round(prediction[0])}")
		return {"prediction": round(prediction[0])}


@backend_app.post("/predict-newsletter")
def predict_newsletter(data: NewsletterInputData):
	# map categorical data for one-hot encoding
	is_high_roller, is_luxury_estate, is_wine_enthusiast = encode_customer_segment(data.CustomerSegment)
	(is_east_south_central, is_middle_atlantic, is_mountain,
	  is_new_england, is_pacific, is_south_atlantic,
		is_west_north_central, is_west_south_central) = encode_division(data.Division)
	
	# config input data for model prediction
	input_data = [[data.OrderVolume, data.SaleAmount, 
				   is_high_roller, is_luxury_estate, is_wine_enthusiast, 
				   is_east_south_central, is_middle_atlantic, is_mountain,
				   is_new_england, is_pacific, is_south_atlantic,
					is_west_north_central, is_west_south_central,
					data.WinemakerCallSubscr, data.EmailSubscr]]
	
	newsletter_model, newsletter_model_type = newsletter_model_tuple
	if newsletter_model_type == "Logit":
		input_data[0].insert(0, 1.0)
	
	prediction = newsletter_model.predict(input_data)

	if newsletter_model_type == "RF":
		logging.info(f"RF Newsletter Prediction: {prediction}")
		return {"prediction": int(prediction[0])}
	else:
		logging.info(f"Logit Newsletter Probability: {prediction} and Prediction: {round(prediction[0])}")
		return {"prediction": round(prediction[0])}
		

@backend_app.post("/predict-winemaker")
def predict_winemaker(data: WinemakerInputData):
	# map categorical data for one-hot encoding
	is_high_roller, is_luxury_estate, is_wine_enthusiast = encode_customer_segment(data.CustomerSegment)
	(is_east_south_central, is_middle_atlantic, is_mountain,
	  is_new_england, is_pacific, is_south_atlantic,
		is_west_north_central, is_west_south_central) = encode_division(data.Division)
	
	# config input data for model prediction
	input_data = [[data.OrderVolume, data.SaleAmount, 
				   is_high_roller, is_luxury_estate, is_wine_enthusiast, 
				   is_east_south_central, is_middle_atlantic, is_mountain,
				   is_new_england, is_pacific, is_south_atlantic,
					is_west_north_central, is_west_south_central,
					data.NewsletterSubscr, data.EmailSubscr]]
	
	winemaker_model, winemaker_model_type = winemaker_model_tuple
	if winemaker_model_type == "Logit":
		input_data[0].insert(0, 1.0)
	
	prediction = winemaker_model.predict(input_data)
	if winemaker_model_type == "RF":
		logging.info(f"RF Winemaker Prediction: {prediction}")
		return {"prediction": int(prediction[0])}
	else:
		logging.info(f"Logit Winemaker Probability: {prediction} and Prediction: {round(prediction[0])}")
		return {"prediction": round(prediction[0])}



if __name__ == "__main__":
	import uvicorn
	uvicorn.run(backend_app, host="0.0.0.0", port=8000)