import bentoml
import pandas as pd
import numpy as np
from pydantic import BaseModel

# Schéma d'entrée pour la validation
class AccidentInput(BaseModel):
    place: int
    catu: int
    sexe: int
    secu1: float
    year_acc: int
    victim_age: int
    catv: int
    obsm: int
    motor: int
    catr: int
    circ: int
    surf: int
    situ: int
    vma: int
    jour: int
    mois: int
    lum: int
    dep: int
    com: int
    agg_: int
    int: int
    atm: int
    col: int
    lat: float
    long: float
    hour: int
    nb_victim: int
    nb_vehicules: int

# 1. Service d'Inférence (Modèle)
@bentoml.service(name="accident_inference_service")
class AccidentInferenceService:
    # Récupération du modèle scikit-learn
    model = bentoml.sklearn.get("accidents_rf:latest")

    @bentoml.api
    def run_inference(self, input_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(input_df)

# 2. Service API (Orchestrateur)
@bentoml.service(name="accident_api_service")
class AccidentApiService:
    # On définit la dépendance vers le service d'inférence
    inference_service = bentoml.depends(AccidentInferenceService)

    @bentoml.api
    def predict(self, input_data: AccidentInput) -> dict:
        # Validation et conversion via Pydantic/Pandas
        input_df = pd.DataFrame([input_data.model_dump()])
        
        # Appel du service d'inférence via la dépendance
        prediction = self.inference_service.run_inference(input_df)
        
        return {"prediction": prediction.tolist()}
