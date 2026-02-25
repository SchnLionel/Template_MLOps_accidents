import bentoml
import numpy as np
import pandas as pd

def make_prediction(model_name: str, input_df: pd.DataFrame) -> np.ndarray:
    """
    Load a model from BentoML model store and make a prediction.
    """
    # On charge le modèle scikit-learn enregistré dans BentoML
    model = bentoml.sklearn.load_model(model_name)
    return model.predict(input_df)
