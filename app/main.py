from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
from typing import Any, Optional, List
import joblib
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
import io


features = ["OverallQual",
"GrLivArea",
"OverallCond",
"TotalBsmtSF",
"GarageCars",
"YearBuilt",
"SaleCondition",
"LotArea",
"CentralAir",
"BsmtFinSF1",
"Functional",
"Fireplaces",
"YearRemodAdd",
"SecondFlrSF",
"KitchenQual",
"PavedDrive",
"BsmtQual",
"ScreenPorch",
"WoodDeckSF",
"BsmtFullBath"]

class HouseDataInputSchema(BaseModel):
    OverallQual: Optional[str]
    GrLivArea: Optional[str]
    OverallCond: Optional[str]
    TotalBsmtSF: Optional[str]
    GarageCars: Optional[str]
    YearBuilt: Optional[str]
    SaleCondition: Optional[str]
    LotArea: Optional[str]
    CentralAir: Optional[str]
    BsmtFinSF1: Optional[str]
    Functional: Optional[str]
    Fireplaces: Optional[str]
    YearRemodAdd: Optional[str]
    SecondFlrSF: Optional[str]
    KitchenQual: Optional[str]
    PavedDrive: Optional[str]
    BsmtQual: Optional[str]
    ScreenPorch: Optional[str]
    WoodDeckSF: Optional[str]
    BsmtFullBath: Optional[str]


class MultipleHouseDataInputs(BaseModel):
    inputs: List[HouseDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "OverallQual": 5,
                        "GrLivArea": 896,
                        "OverallCond": 6,
                        "TotalBsmtSF": 882.0,
                        "GarageCars": 1.0,
                        "YearBuilt": 1961,
                        "SaleCondition": "Normal",
                        "LotArea": 11622,
                        "CentralAir": "Y",
                        "BsmtFinSF1": 468.0,
                        "Functional": "Typ",
                        "Fireplaces": 0,
                        "YearRemodAdd": 1961,
                        "SecondFlrSF": 0,
                        "KitchenQual": "TA",
                        "PavedDrive": "Y",
                        "BsmtQual": "TA",
                        "ScreenPorch": 120,
                        "WoodDeckSF": 140,
                        "BsmtFullBath": 0.0,
                    }
                ]
            }
        }


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the House Price Prediction API! You can go ahead to make request at /single_predict now."}


@app.post("/single_predict")
async def single_predict(input_entry: MultipleHouseDataInputs) -> Any:
    """
    Make house price predictions with the TID regression model
    """
    input_df = pd.DataFrame(jsonable_encoder(input_entry.inputs))
    model_filename = "/app/house_price_pipe.pkl"
    with open(model_filename, 'rb') as fo:
        trained_model = joblib.load(fo)
    predictions = trained_model.predict(input_df)
    input_df["predictions"] = np.exp(predictions)
    return input_df.to_json()

@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)) -> Any:
    """
    Make house price predictions with the TID regression model by accepting a csv file
    """
    filename = file.filename
    file_extension = filename.split(".")[-1] == "csv"
    if not file_extension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")

    input_df = pd.read_csv(file.file)
    original_cols = input_df.columns
    input_df = input_df.rename(columns={"2ndFlrSF": "SecondFlrSF"})
    model_filename = "/app/house_price_pipe.pkl"
    with open(model_filename, 'rb') as fo:
        trained_model = joblib.load(fo)
    predictions = trained_model.predict(input_df[features])
    input_df.columns = original_cols
    input_df["predictions"] = np.exp(predictions)

    stream = io.StringIO()
    input_df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv"
                                 )

    response.headers["Content-Disposition"] = f"attachment; filename={filename}_predictions.csv"
    return response
    # return input_df.to_json()


# if __name__ == '__main__':
#     import uvicorn
#
#     uvicorn.run(app, host="localhost", port=8001, log_level="debug")