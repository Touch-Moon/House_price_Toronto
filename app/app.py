import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# Initialize FastAPI app
app = FastAPI()

# Basic route to check if server is up and running
@app.get("/")
def read_root():
    return {"message": "Hello World"}

# Configure logging for debugging purposes
logging.basicConfig(level=logging.INFO)

# Load the trained model
model_path = "models/random_forest_model.pkl"
model = joblib.load(model_path)

# Get expected feature names from the trained model
expected_features = model.feature_names_in_

# Define request model for input data validation
class HouseFeatures(BaseModel):
    sqft: float
    bedrooms_ag: int
    bedrooms_bg: int
    bathrooms: int
    parking: int
    mean_district_income: float

# Prediction route to receive input data and make house price prediction
@app.post("/predict")
async def predict_house_price(data: HouseFeatures):
    try:
        input_data = data.dict()
        logging.info(f"🔍 Received input: {input_data}")  # Log the input data for debugging

        # Create a DataFrame with the input data
        input_df = pd.DataFrame([input_data])

        # Reindex the columns to match the model's expected features
        input_df = input_df.reindex(columns=expected_features, fill_value=0)

        logging.info(f"📊 Processed DataFrame for prediction:\n{input_df}")  # Log the processed DataFrame

        # Predict the house price using the model
        predicted_price = model.predict(input_df)[0]

        # If the predicted price is too low, apply exponential transformation
        if predicted_price < 20:
            predicted_price = np.exp(predicted_price)

        logging.info(f"💰 Predicted price: {predicted_price}")  # Log the predicted price

        return {"predicted_price": predicted_price}

    except Exception as e:
        logging.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")