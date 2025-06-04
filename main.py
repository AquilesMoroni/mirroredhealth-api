from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Carrega o modelo treinado oiiiiiiiiiiiiiiiii
model = joblib.load("model.joblib")

# Exemplo de campos esperados - substitua pelos reais conforme seu modelo
class ModelInput(BaseModel):
    Gender: int
    Age: int
    Helpful_for_studying: int
    Daily_usages: int
    Performance_impact: int
    Usage_distraction: int
    Attention_span: int
    Useful_features: int
    Beneficial_subject: int
    Usage_symptoms: int
    Symptom_frequency: int
    Health_precautions: int
    Mobile_phone_use_for_education: int
    Health_rating: int

@app.post("/predict")
def predict(data: ModelInput):
    try:
        # Converte a entrada em DataFrame com uma linha
        input_df = pd.DataFrame([data.dict()])

        # Realiza a previsão
        prediction = model.predict(input_df)

        return {"health_risk_prediction": int(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

         # testeee # 
        
