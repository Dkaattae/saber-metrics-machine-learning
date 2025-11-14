import pandas as pd
from typing import Literal
from pydantic import BaseModel, Field
from fastapi import FastAPI
import uvicorn

from predict import predict_single

app = FastAPI(title="MLB-home-team-win-prediction")
ALLOWED_DAYOFWEEK = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
ALLOWED_PARK_IDS = ['ANA01', 'ARL03', 'ATL03', 'BAL12', 'BIR01', 'BOS07',
        'CHI11', 'CHI12', 'CIN09', 'CLE08', 'DEN02', 'DET05', 'HOU03',
        'KAN06', 'LON01', 'LOS03', 'MEX02', 'MIA02', 'MIL06', 'MIN04',
        'NYC20', 'NYC21', 'OAK01', 'PHI13', 'PHO01', 'PIT08', 'SAN02',
        'SEA03', 'SEO01', 'SFO03', 'STL10', 'STP01', 'TOR02', 'WAS11', 'WIL02']

class Game(BaseModel):
    date: int = Field(..., ge=10**7)
    dayofweek: Literal[*ALLOWED_DAYOFWEEK]
    away_league: Literal['AL', 'NL']
    home_league: Literal['AL', 'NL']
    park_id: Literal[*ALLOWED_PARK_IDS]
    home_OPS_blend: float = Field(..., ge=0.0)
    home_FIP_blend: float = Field(..., ge=0.0)
    home_FPCT_blend: float = Field(..., ge=0.0)
    away_OPS_blend: float = Field(..., ge=0.0)
    away_FIP_blend: float = Field(..., ge=0.0)
    away_FPCT_blend: float = Field(..., ge=0.0)

class PredictResponse(BaseModel):
    win_prob: float = Field(..., ge=0.0, le=1.0)
    predicted_win: bool
    predicted_lose: bool

@app.post("/predict")
def predict(game: Game) -> PredictResponse:
    data_dict = game.model_dump()
    
    df = pd.DataFrame([data_dict])
    prediction = predict_single(df)

    return PredictResponse(**prediction)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
