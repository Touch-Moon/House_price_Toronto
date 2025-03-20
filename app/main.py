from fastapi import FastAPI
import joblib

# Create a FastAPI instance
app = FastAPI()

# Load the pre-trained model from the specified location
model = joblib.load('/home/ubuntu/models/random_forest_model.pkl')

# Example endpoint to check if the server is running
@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}

# Endpoint to handle prediction requests
@app.post("/predict/")
def predict(data: dict):
    # Process the input data and prepare it for the model
    X_input = [data["features"]]  # Adjust this based on how your model expects input data
    prediction = model.predict(X_input)  # Make a prediction using the model
    return {"prediction": prediction.tolist()}  # Return the prediction result as a list