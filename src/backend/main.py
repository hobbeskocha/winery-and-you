import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import logging
# import pandas as pd
from google.cloud import storage

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

metrics_file = download_blob(bucket_name, "model-metrics.txt", "/tmp/model-metrics.txt")
print(f"Metrics file downloaded to /tmp/model-metrics.txt")
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

	email_model_metrics = [model.split(":")[1] for model in email_models]
	email_model_accuracies = [model.split(";")[0] for model in email_model_metrics]
	best_email_model = (model_log_email, "Logit") if email_model_accuracies[0] > email_model_accuracies[1] else (model_rf_email, "RF")

	newsletter_model_metrics = [model.split(":")[1] for model in newsletter_models]
	newsletter_model_accuracies = [model.split(";")[0] for model in newsletter_model_metrics]
	best_newsletter_model = (model_log_newsletter, "Logit") if newsletter_model_accuracies[0] > newsletter_model_accuracies[1] else (model_rf_newsletter, "RF")

	winemaker_model_metrics = [model.split(":")[1] for model in winemaker_models]
	winemaker_model_accuracies = [model.split(";")[0] for model in winemaker_model_metrics]
	best_winemaker_model = (model_log_winemaker, "Logit") if winemaker_model_accuracies[0] > winemaker_model_accuracies[1] else (model_rf_winemaker, "RF")
	
	return best_email_model, best_newsletter_model, best_winemaker_model

def get_scaling_factors():
	# sale scale mean, order scale mean
	metrics = metrics_line.split(",")
	email_metrics = [metric for metric in metrics if "Email" in metric]
	newsletter_metrics = [metric for metric in metrics if "Newsletter" in metric]
	winemaker_metrics = [metric for metric in metrics if "Winemaker" in metric]

	email_metrics = [metric.split(":")[1] for metric in email_metrics]
	newsletter_metrics = [metric.split(":")[1] for metric in newsletter_metrics]
	winemaker_metrics = [metric.split(":")[1] for metric in winemaker_metrics]

	email_metrics = [metric.split(";") for metric in email_metrics]
	newsletter_metrics = [metric.split(";") for metric in newsletter_metrics]
	winemaker_metrics = [metric.split(";") for metric in winemaker_metrics]

	email_factors = email_metrics[0][1:]
	newsletter_factors = newsletter_metrics[0][1:]
	winemaker_factors = winemaker_metrics[0][1:]

	email_sale_scale, email_sale_mean = float(email_factors[0]), float(email_factors[1])
	email_order_scale, email_order_mean = float(email_factors[2]), float(email_factors[3])
	newsletter_sale_scale, newsletter_sale_mean = float(newsletter_factors[0]), float(newsletter_factors[1])
	newsletter_order_scale, newsletter_order_mean = float(newsletter_factors[2]), float(newsletter_factors[3])
	winemaker_sale_scale, winemaker_sale_mean = float(winemaker_factors[0]), float(winemaker_factors[1])
	winemaker_order_scale, winemaker_order_mean = float(winemaker_factors[2]), float(winemaker_factors[3])

	return (email_sale_scale, email_sale_mean, email_order_scale, email_order_mean,
			newsletter_sale_scale, newsletter_sale_mean, newsletter_order_scale, newsletter_order_mean,
			winemaker_sale_scale, winemaker_sale_mean, winemaker_order_scale, winemaker_order_mean)

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
	(email_sale_scale, email_sale_mean, email_order_scale, email_order_mean,
	 newsletter_sale_scale, newsletter_sale_mean, newsletter_order_scale, newsletter_order_mean,
	 winemaker_sale_scale, winemaker_sale_mean, winemaker_order_scale, winemaker_order_mean) = get_scaling_factors()

	if channel == "Email":
		input_data[-1].append(data.NewsletterSubscr)
		input_data[-1].append(data.WinemakerCallSubscr)
		# scale based on Email training data scale and mean
		input_data[0][0] = (data.OrderVolume - email_order_mean) / email_order_scale
		input_data[0][1] = (data.SaleAmount - email_sale_mean) / email_sale_scale

	elif channel == "Newsletter":
		input_data[-1].append(data.WinemakerCallSubscr)
		input_data[-1].append(data.EmailSubscr)
		# scale based on Newsletter training data scale and mean
		input_data[0][0] = (data.OrderVolume - newsletter_order_mean) / newsletter_order_scale
		input_data[0][1] = (data.SaleAmount - newsletter_sale_mean) / newsletter_sale_scale

	elif channel == "Winemaker":
		input_data[-1].append(data.NewsletterSubscr)
		input_data[-1].append(data.EmailSubscr)
		# scale based on Winemaker training data scale and mean
		input_data[0][0] = (data.OrderVolume - winemaker_order_mean) / winemaker_order_scale
		input_data[0][1] = (data.SaleAmount - winemaker_sale_mean) / winemaker_sale_scale
	
	# if model_type == "RF":
	# 	input_data = pd.DataFrame(input_data, columns=model.feature_names_in_)

	if model_type == "Logit":
		input_data[0].insert(0, 1.0)
	
	print(input_data)
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
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))