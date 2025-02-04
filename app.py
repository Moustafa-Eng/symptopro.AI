import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn

# Define the input schema
class InputData(BaseModel):
    data: list

# Load the ML model
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    raise Exception("Model file not found. Ensure 'model.pkl' is in the same directory.")
except Exception as e:
    raise Exception(f"Error loading model: {e}")

# Disease map
disease_map = {
    0: "Paroxysmal Positional Vertigo",
    1: "AIDS",
    2: "Acne",
    3: "Alcoholic hepatitis",
    4: "Allergy",
    5: "Arthritis",
    6: "Bronchial Asthma",
    7: "Cervical spondylosis",
    8: "Chicken pox",
    9: "Chronic cholestasis",
    10: "Common Cold",
    11: "Dengue",
    12: "Diabetes",
    13: "Dimorphic hemorrhoids (piles)",
    14: "Drug Reaction",
    15: "Fungal infection",
    16: "GERD",
    17: "Gastroenteritis",
    18: "Heart attack",
    19: "Hepatitis B",
    20: "Hepatitis C",
    21: "Hepatitis D",
    22: "Hepatitis E",
    23: "Hypertension",
    24: "Hyperthyroidism",
    25: "Hypoglycemia",
    26: "Hypothyroidism",
    27: "Impetigo",
    28: "Jaundice",
    29: "Malaria",
    30: "Migraine",
    31: "Osteoarthritis",
    32: "Paralysis (brain hemorrhage)",
    33: "Peptic ulcer disease",
    34: "Pneumonia",
    35: "Psoriasis",
    36: "Tuberculosis",
    37: "Typhoid",
    38: "Urinary tract infection",
    39: "Varicose veins",
    40: "Hepatitis A",
}

# Initialize the FastAPI application
app = FastAPI()

@app.post("/predict")
async def predict(input_data: InputData):
    # Validate input length
    if len(input_data.data) != 132:
        raise HTTPException(status_code=400, detail="Input data must contain exactly 132 numbers.")

    # Validate input values
    if any(x not in [0, 1] for x in input_data.data):
        raise HTTPException(status_code=400, detail="Input data must contain only 0s and 1s.")

    # Convert input to numpy array
    input_array = np.array(input_data.data).reshape(1, -1)  

    # Perform prediction
    try:
        prediction = model.predict(input_array)[0]  # Get the first prediction (numpy.int64)
        prediction = int(prediction)  # Convert numpy.int64 to regular int

        # Get disease name from the map
        disease_name = disease_map.get(prediction, "Unknown Disease")

        return {"Disease": disease_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

@app.get("/")
def root():
    return {"message": "Welcome FADY to the ML Prediction API!"}
  
  
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Get PORT from environment, default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)  # Use 0.0.0.0