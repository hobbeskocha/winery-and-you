import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import logging
# import pandas as pd
from google.cloud import storage
import uvicorn

from classes.EmailInputData import EmailInputData
from classes.NewsletterInputData import NewsletterInputData
from classes.WinemakerInputData import WinemakerInputData
from classes.InputData import InputData
from helpers.categorical_encoders import *

# -------------------------Utilities-------------------------

# # Load models from local storage
# model_log_email = joblib.load("../model-artifacts/log_email.pkl")
# model_log_newsletter = joblib.load("../model-artifacts/log_newsletter.pkl")
# model_log_winemaker = joblib.load("../model-artifacts/log_winemaker.pkl")
# model_rf_email = joblib.load("../model-artifacts/rf_email.pkl")
# model_rf_newsletter = joblib.load("../model-artifacts/rf_newsletter.pkl")
# model_rf_winemaker = joblib.load("../model-artifacts/rf_winemaker.pkl")

# metrics_file = open("../model-artifacts/model-metrics.txt")    
# metrics_line = metrics_file.readline()
# metrics_file.close()

# Load models from GCS
def download_blob(bucket_name, source_blob_name, destination_file_name):
	"""Downloads a blob from the bucket."""
	storage_client = storage.Client()
	bucket = storage_client.bucket(bucket_name)
	blob = bucket.blob(source_blob_name)
	blob.download_to_filename(destination_file_name)
	print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

bucket_name = "winery-ml-models-bucket"
model_artifacts = {
	"log_email": "log_email.pkl",
	"log_newsletter": "log_newsletter.pkl",
	"log_winemaker": "log_winemaker.pkl",
	"rf_email": "rf_email.pkl",	
	"rf_newsletter": "rf_newsletter.pkl",
	"rf_winemaker": "rf_winemaker.pkl"
}

tmp_dir = "/tmp/models"
os.makedirs(tmp_dir, exist_ok=True)
models = {}
for model_name, gcs_path in model_artifacts.items():
	local_path = os.path.join(tmp_dir, model_name + ".pkl")
	download_blob(bucket_name, gcs_path, local_path)
	models[model_name] = joblib.load(local_path)

model_log_email, model_log_newsletter, model_log_winemaker = models["log_email"], models["log_newsletter"], models["log_winemaker"]
model_rf_email, model_rf_newsletter, model_rf_winemaker = models["rf_email"], models["rf_newsletter"], models["rf_winemaker"]

metrics_file = download_blob(bucket_name, "/model-metrics.txt", "/tmp/model-metrics.txt")
metrics_file = open("/tmp/model-metrics.txt")    
metrics_line = metrics_file.readline()
metrics_file.close()

def model_selector():
	"""
	compare model accuracies and return best model and model type as tuple
	"""
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

def prepare_input(data: InputData, model_tuple: tuple, channel: str): 
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
					is_west_north_central, is_west_south_central]]
	
	model, model_type = model_tuple
	if model_type == "Logit":
		input_data[0].insert(0, 1.0)

	if channel == "Email":
		input_data[-1].append(data.NewsletterSubscr)
		input_data[-1].append(data.WinemakerCallSubscr)
	elif channel == "Newsletter":
		input_data[-1].append(data.WinemakerCallSubscr)
		input_data[-1].append(data.EmailSubscr)
	elif channel == "Winemaker":
		input_data[-1].append(data.NewsletterSubscr)
		input_data[-1].append(data.EmailSubscr)
	
	# if model_type == "RF":
	# 	input_data = pd.DataFrame(input_data, columns=model.feature_names_in_)
	
	return input_data, model, model_type

# -------------------------Backend App-------------------------

# Select best models
email_model_tuple, newsletter_model_tuple, winemaker_model_tuple = model_selector()

# Create backend app
app = FastAPI()
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# HTTP requests
@app.get("/")
def read_root():
	return {"message": "Hello World"}

@app.get("/version")
def read_version():
	return {f"Python version": f"{sys.version}"}

@app.post("/predict-email")
def predict_email(data: EmailInputData):
	input_data, email_model, email_model_type = prepare_input(data, email_model_tuple, "Email")
	prediction = email_model.predict(input_data)

	if email_model_type == "RF":
		logging.info(f"RF Email Prediction: {prediction}")
		return {"prediction": int(prediction[0])}
	else:    
		logging.info(f"Logit Email Probability: {prediction} and Prediction: {round(prediction[0])}")
		return {"prediction": round(prediction[0])}


@app.post("/predict-newsletter")
def predict_newsletter(data: NewsletterInputData):
	input_data, newsletter_model, newsletter_model_type = prepare_input(data, newsletter_model_tuple, "Newsletter")
	prediction = newsletter_model.predict(input_data)

	if newsletter_model_type == "RF":
		logging.info(f"RF Newsletter Prediction: {prediction}")
		return {"prediction": int(prediction[0])}
	else:
		logging.info(f"Logit Newsletter Probability: {prediction} and Prediction: {round(prediction[0])}")
		return {"prediction": round(prediction[0])}
		

@app.post("/predict-winemaker")
def predict_winemaker(data: WinemakerInputData):
	input_data, winemaker_model, winemaker_model_type = prepare_input(data, winemaker_model_tuple, "Winemaker")
	prediction = winemaker_model.predict(input_data)

	if winemaker_model_type == "RF":
		logging.info(f"RF Winemaker Prediction: {prediction}")
		return {"prediction": int(prediction[0])}
	else:
		logging.info(f"Logit Winemaker Probability: {prediction} and Prediction: {round(prediction[0])}")
		return {"prediction": round(prediction[0])}



if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))