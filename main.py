from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
from io import BytesIO
import numpy as np

app = FastAPI()

class Item(BaseModel):
    year: int
    km_driven: float
    mileage: float
    engine: int
    max_power: int
    torque: float
    max_torque_rpm: float
    seats: float


class Items(BaseModel):
    objects: List[Item]

@app.post("/predict_item")
def predict_item(item: Item) -> float:

    result = []
    weights = pd.read_pickle('data.pickle0').iloc[0].to_list()
    result.append(weights[0])

    count = 1
    for key, value in item.__dict__.items():
        result.append(weights[count] * value)
        count += 1
    return sum(result)

@app.post("/predict_items")
def predict_items(file: UploadFile = File()) -> FileResponse:
    weights = pd.read_pickle('data.pickle0').iloc[0].to_list()
    content = file.file.read()
    buffer = BytesIO(content)
    df = pd.read_csv(buffer).to_numpy()

    inter = weights[0]
    weights = np.array(weights[1:])
    weights = weights.reshape(8, 1)

    res = df @ weights + inter
    df = pd.DataFrame(df)
    new_df = pd.DataFrame(res)
    df = df.merge(new_df, left_index=True, right_index=True)
    df.to_csv('result.csv')
    response = FileResponse(path='result.csv', media_type='text/csv')
    return response