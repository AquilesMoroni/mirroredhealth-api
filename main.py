from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import json
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse

# Cria a instância do FastAPI
app = FastAPI()

# Configura o CORS para permitir requisições de qualquer origem (sites ou apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega o modelo treinado de IA
model = joblib.load("modelo_treinado.pkl")

# Carrega as colunas usadas no treinamento da IA
model_columns = joblib.load("model_columns.joblib")

# Campos que a API deve receber obrigatoriamente (/predict)
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
        print("Recebido:", data.dict())
        # Verifica se os nomes dos campos batem com os nomes das colunas usados no modelo treinado
        mapping = {
            "Gender": "Gender",
            "Age": "Age",
            "Helpful_for_studying": "Helpful for studying",
            "Daily_usages": "Daily usages",
            "Performance_impact": "Performance impact",
            "Usage_distraction": "Usage distraction",
            "Attention_span": "Attention span",
            "Useful_features": "Useful features",
            "Beneficial_subject": "Beneficial subject",
            "Usage_symptoms": "Usage symptoms",
            "Symptom_frequency": "Symptom frequency",
            "Health_precautions": "Health precautions",
            "Mobile_phone_use_for_education": "Mobile phone use for education",
            "Health_rating": "Health rating"
        }
        data_dict = data.dict()
        data_renamed = {mapping[k]: v for k, v in data_dict.items()}
        input_df = pd.DataFrame([data_renamed])
        # Garante que todas as colunas do modelo estejam presentes no DataFrame
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_columns]
        print("DataFrame para predição:", input_df)
        # Faz a predição usando o modelo de IA treinado, para indicar a % em que o aluno é afetado
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]
        return {
            "health_risk_prediction": int(prediction[0]),
            "probability": round(float(probability) * 100, 2)  # porcentagem
        }
    except Exception as e:
        print("Erro:", e)
        raise HTTPException(status_code=500, detail=str(e)) 