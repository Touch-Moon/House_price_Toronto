import joblib
import numpy as np
import pandas as pd

# Load the trained model
model_path = "models/random_forest_model.pkl"
model = joblib.load(model_path)

# 🔹 모델이 학습한 feature 이름 확인
expected_features = model.feature_names_in_

# 🔹 예측을 위한 데이터 준비 (맞는 feature 순서대로)
test_data = {
    "sqft": [1000],
    "bedrooms_ag": [2],
    "bedrooms_bg": [1],
    "bathrooms": [2],
    "parking": [1],
    "mean_district_income": [75000]
}

# 🔹 누락된 feature 채우기 (기본값 0)
for feature in expected_features:
    if feature not in test_data:
        test_data[feature] = [0]

# 🔹 DataFrame 변환
test_input = pd.DataFrame(test_data)[expected_features]

# 🔹 Feature 개수 확인
print(f"✅ Input Shape: {test_input.shape}")  # feature 개수 확인
print(f"✅ Model Feature Count: {len(expected_features)}")  # 모델이 학습한 feature 개수

# 🔹 예측 수행
predicted_price = model.predict(test_input)[0]

# 🔹 로그 변환 확인 후 역변환
if predicted_price < 20:  # log 변환된 데이터일 가능성
    predicted_price = np.exp(predicted_price)
    print(f"🔄 Log-transformed price detected, applying exp()")

print(f"🏡 Predicted Price: ${predicted_price}")