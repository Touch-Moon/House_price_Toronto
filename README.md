# House Price Prediction Model

This project demonstrates a house price prediction model using Random Forest, built with Python, FastAPI, and deployed on AWS EC2. The model is trained using housing data from Toronto and can predict house prices based on various features such as square footage, number of bedrooms, bathrooms, parking spaces, and district income.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [API Deployment](#api-deployment)
- [License](#license)

## Project Overview

This project is aimed at building a predictive model for house prices based on several features. The model is trained using the Random Forest algorithm and exposed as an API using FastAPI. The API allows users to input feature values, and the model predicts the house price based on those inputs.

Key Features:
- Random Forest model for price prediction.
- API built with FastAPI for easy integration and scalability.
- Deployed on AWS EC2 for production usage.

## Installation

To run this project locally, follow these steps:

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/House_price_Toronto.git
    cd House_price_Toronto
    ```

2. Create a virtual environment:
    ```bash
    python3 -m venv fastapi_env
    source fastapi_env/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. If you don't have the model file (`random_forest_model.pkl`), make sure to upload it into the `/models/` directory.

## Usage

1. To start the FastAPI application:
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```

2. Access the API documentation at:
    - `http://127.0.0.1:8000/docs`

3. Send a POST request to `/predict` with the following body to get predictions:
    ```json
    {
        "sqft": 1000,
        "bedrooms": 3,
        "bathrooms": 2,
        "parking": 1,
        "mean_district_income": 75000
    }
    ```

## Model Training

The model is trained using housing data (e.g., square footage, number of bedrooms, etc.) with a Random Forest algorithm. You can train your own model by running the `train_model.py` script.

```bash
python train_model.py
