import bentoml
import numpy as np

# On récupère le modèle que vous venez d'enregistrer
model_ref = bentoml.sklearn.get("accidents_rf:latest")

@bentoml.service(name="rf_classifier_service")
class RFClassifierService:
    model = model_ref

    @bentoml.api
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        return self.model.predict(input_data)
