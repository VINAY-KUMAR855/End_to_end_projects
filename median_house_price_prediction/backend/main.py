from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from custom_transformers import column_ratio, ratio_name, ClusterSimilarity
import sys

# register above functions/class onto sys.modules['__main__']
for _mod_name in ("__main__", "__mp_main__"):
    if _mod_name in sys.modules:
        sys.modules[_mod_name].column_ratio = column_ratio
        sys.modules[_mod_name].ratio_name = ratio_name
        sys.modules[_mod_name].ClusterSimilarity = ClusterSimilarity


# load model
model = joblib.load(
    "model/my_california_housing_model.pkl"
)

# create FastAPI app
app = FastAPI()

# Allow frontend (HTML/JS) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# input format
class HousingData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

@app.get("/")
def home():
    return {
        "message": "California Housing Price Predictor"
    }

@app.post("/predict")
def predict(data: HousingData):
    # convert input into dataframe
    input_data = pd.DataFrame([data.dict()])
    prediction = model.predict(input_data)
    return {
        "predicted_price": float(prediction[0])
    }
