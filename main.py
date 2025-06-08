from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import json
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ou especifique ["http://localhost"] para mais segurança
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega o modelo treinado 
model = joblib.load("model.joblib")
# Carrega as colunas usadas no treinamento
model_columns = joblib.load("model_columns.joblib")  # Salve isso no seu notebook!

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

@app.get("/metrics")
def get_metrics():
    try:
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
        return JSONResponse(content=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
#Exibir os graficos do CSV na tela/página de GRAFICO.HTML
@app.get("/grafico/{nome}")
def get_grafico(nome: str):
    caminho = f"./{nome}"  # Usa o nome passado na URL
    return FileResponse(caminho)

@app.post("/predict")
def predict(data: ModelInput):
    try:
        print("Recebido:", data.dict())
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
        # Garante que todas as colunas do modelo estejam presentes
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_columns]
        print("DataFrame para predição:", input_df)
        prediction = model.predict(input_df)
        return {"health_risk_prediction": int(prediction[0])}
    except Exception as e:
        print("Erro:", e)
        raise HTTPException(status_code=500, detail=str(e))