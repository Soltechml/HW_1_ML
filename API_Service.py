#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import pandas as pd
import io
import joblib

app = FastAPI()

# Загрузка обученных объектов

model = joblib.load('saved_model.pkl')
encoder = joblib.load('saved_encoder.pkl')
scaler = joblib.load('saved_scaler.pkl')

class Item(BaseModel):
    # Определение полей объекта
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: int

class Items(BaseModel):
    # Класс для коллекции объектов
    objects: List[Item]

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    # Преобразование данных в DataFrame
    data = pd.DataFrame([item.dict()])

    # Кодирование категориальных признаков
    encoded_data = encoder.transform(data[categorical_cols_with_seats])

    # Масштабирование числовых признаков
    data[numeric_cols] = scaler.transform(data[numeric_cols])

    # Объединение данных
    data_prepared = pd.concat([data.drop(categorical_cols_with_seats, axis=1), 
                               pd.DataFrame(encoded_data, columns=encoder.get_feature_names(categorical_cols_with_seats))], axis=1)

    # Предсказание
    prediction = model.predict(data_prepared)[0]
    return prediction

@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)) -> str:
    # Чтение CSV-файла
    df = pd.read_csv(io.StringIO(str(file.file.read(), 'utf-8')))

    # Кодирование категориальных признаков
    encoded_data = encoder.transform(df[categorical_cols_with_seats])

    # Масштабирование числовых признаков
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Объединение данных
    data_prepared = pd.concat([df.drop(categorical_cols_with_seats, axis=1), 
                               pd.DataFrame(encoded_data, columns=encoder.get_feature_names(categorical_cols_with_seats))], axis=1)

    # Предсказание
    predictions = model.predict(data_prepared)

    # Добавление предсказаний в DataFrame и сохранение результата
    df['predictions'] = predictions
    result_file = io.StringIO()
    df.to_csv(result_file, index=False)
    result_file.seek(0)

    return result_file.read()

