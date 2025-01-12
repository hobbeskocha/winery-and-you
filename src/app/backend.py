from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model_log_email = joblib.load("../model-artifacts/log_email.pkl")

backend_app = FastAPI()

class InputData(BaseModel):
    OrderVolume: int
    CustomerSegment: str # TODO: one-hot encode
    Division: str # TODO: one-hot encode
    SaleAmount: float
    NewsletterSubscr: bool
    WinemakerCallSubscr: bool 

@backend_app.post("/predict")
def predict(data: InputData):
    input_data = [[data.OrderVolume, data.CustomerSegment,
                    data.Division, data.SaleAmount,
                      data.NewsletterSubscr, data.WinemakerCallSubscr]] # TODO: one-hot encode categorical variables
    prediction = model_log_email.predict(input_data)
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(backend_app, host="0.0.0.0", port=8000)