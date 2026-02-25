# src/service.py
import numpy as np
import bentoml
from pydantic import BaseModel, Field, ConfigDict
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
from datetime import datetime, timedelta

# Ne pas hardcoder les secrets en production
JWT_SECRET_KEY = "your_jwt_secret_key_here"
JWT_ALGORITHM = "HS256"

USERS = {
    "user123": "password123",
    "user456": "password456",
}

class Credentials(BaseModel):
    username: str
    password: str

class InputModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

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
    int_: int = Field(alias="int")
    atm: int
    col: int
    lat: float
    long: float
    hour: int
    nb_victim: int
    nb_vehicules: int

def create_jwt_token(user_id: str) -> str:
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {"sub": user_id, "exp": expiration}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path.endswith("/predict"):
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(status_code=401, content={"detail": "Missing authentication token"})
            try:
                # Bearer <token>
                parts = token.split()
                if len(parts) != 2:
                    return JSONResponse(status_code=401, content={"detail": "Invalid authorization header"})
                token = parts[1]
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token has expired"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})
            request.state.user = payload.get("sub")
        return await call_next(request)

@bentoml.service
class RFModelService:
    def __init__(self):
        self.model = bentoml.sklearn.load_model("accidents_rf:latest")

    @bentoml.api
    def predict_array(self, features: list[float]) -> list[float]:
        x = np.array(features, dtype=float).reshape(1, -1)
        pred = self.model.predict(x)
        return pred.tolist()

@bentoml.service
class RFClassifierService:
    model_service = bentoml.depends(RFModelService)

    @bentoml.api(route="/login")
    def login(self, credentials: Credentials):
        if USERS.get(credentials.username) == credentials.password:
            token = create_jwt_token(credentials.username)
            return {"token": token}
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})

    @bentoml.api(route="/predict")
    def predict(self, input_data: InputModel):
        features = [
            input_data.place, input_data.catu, input_data.sexe, input_data.secu1,
            input_data.year_acc, input_data.victim_age, input_data.catv, input_data.obsm,
            input_data.motor, input_data.catr, input_data.circ, input_data.surf,
            input_data.situ, input_data.vma, input_data.jour, input_data.mois,
            input_data.lum, input_data.dep, input_data.com, input_data.agg_,
            input_data.int_, input_data.atm, input_data.col, input_data.lat,
            input_data.long, input_data.hour, input_data.nb_victim, input_data.nb_vehicules,
        ]
        pred = self.model_service.predict_array(features)
        return {"prediction": pred}

# Appliquer le middleware sur tout le service
RFClassifierService.add_asgi_middleware(JWTAuthMiddleware)
