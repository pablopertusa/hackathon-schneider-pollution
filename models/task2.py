import polars as pl
import numpy as np
from datetime import datetime, timezone
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import json
from astral.sun import sun
from astral import LocationInfo

def set_seed():
    np.random.seed(27)

set_seed()

def is_daytime(lat, lon, timestamp):
    location = LocationInfo(latitude=lat, longitude=lon)
    date = datetime(timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second, tzinfo=timezone.utc)
    s = sun(location.observer, date=date, tzinfo=timezone.utc)
    return int(s['sunrise'] <= date <= s['sunset'])

def obtain_train_data(pollutant_data: pl.DataFrame, measurement_data: pl.DataFrame, instrument_data: pl.DataFrame, station_code: int, 
                      pollutant_name: str, scaler_x: MinMaxScaler, scaler_y: MinMaxScaler):

    measurement_data = measurement_data.join(instrument_data, on=["Measurement date", "Station code"], how="inner")
    normal_data = measurement_data.filter(pl.col("Instrument status") == 0)
    item_code = pollutant_data.filter(pl.col("Item name") == pollutant_name)["Item code"].to_list()[0]

    df = (
        normal_data
        .with_columns(
            pl.col("Measurement date").str.to_datetime()
        )
        .with_columns(
            pl.col("Measurement date").dt.day().alias("day"),
            pl.col("Measurement date").dt.month().alias("month"),
            pl.col("Measurement date").dt.year().alias("year"),
            pl.col("Measurement date").dt.hour().alias("hour"),
            pl.col("Measurement date").dt.weekday().alias("weekday")
        )
        .filter(pl.col("Station code") == station_code)
        .filter(pl.col("Item code") == item_code)
    )
    df = (
        df
        .with_columns(
            pl.struct(["Latitude", "Longitude", "Measurement date"]).map_elements(lambda row: is_daytime(row["Latitude"], row["Longitude"], row["Measurement date"]), return_dtype=int).alias("is_day")
        )
    )

    df = df.with_columns([pl.col(col).cast(pl.Float64) for col in df.columns])


    X = df.select(["day", "month", "year", "hour", "weekday", "is_day"]).to_numpy()
    y = df.select(pollutant_name).to_numpy()

    X_scaled = scaler_x.fit_transform(X)

    y_scaled = scaler_y.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val, scaler_y

def obtain_test_data(start_date: datetime, end_date: datetime):
    date_range = pd.date_range(start=start_date, end=end_date, freq="h")
    future_df = pl.DataFrame({
        "year": date_range.year,
        "month": date_range.month,
        "day": date_range.day,
        "hour": date_range.hour,
    })
    
    scaler_x = MinMaxScaler()
    future_df = future_df.with_columns([pl.col(col).cast(pl.Float64) for col in future_df.columns])
    future_df_scaled = scaler_x.fit_transform(future_df)
    
    return future_df_scaled, date_range

def reverse_scaler(preds: np.array, scaler: MinMaxScaler):

    predictions = scaler.inverse_transform(preds)
    return predictions

def train_boosting_model(X_train, X_val, y_train, y_val, X_test, scaler_y):
    """
    Entrena un modelo de XGBoost en GPU, lo valida y genera predicciones.
    
    Parámetros:
        X_train: np.array, características de entrenamiento
        X_val: np.array, características de validación
        y_train: np.array, variable objetivo de entrenamiento
        y_val: np.array, variable objetivo de validación
        X_test: np.array, características de prueba
        scaler_y: MinMaxScaler, escalador de la variable objetivo
    
    Retorna:
        y_pred_test: np.array, predicciones en los datos de prueba
    """

    # Convertir a DMatrix de XGBoost compatible con GPU
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    # Configuración del modelo
    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 27,
        "device": "cuda"
    }

    model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, "val")], verbose_eval=False)

    # Predicción en validación
    y_pred_val = model.predict(dval)
    mse_val = mean_squared_error(y_val, y_pred_val)
    print(f"Error cuadrático medio en validación: {mse_val:.4f}")

    # Predicción en test
    y_pred_test = model.predict(dtest)
    y_pred_test = reverse_scaler(y_pred_test.reshape(-1, 1), scaler_y)

    return y_pred_test.flatten()



pollutant = pl.read_csv("data/raw/pollutant_data.csv")
measurement = pl.read_csv("data/raw/measurement_data.csv")
instrument = pl.read_csv("data/raw/instrument_data.csv")

future = [
    {
        "Station code" : 205,
        "pollutant name" : "SO2",
        "period" : [datetime(year = 2023, month= 11, day = 1, hour = 0, tzinfo=timezone.utc), datetime(year = 2023, month= 11, day = 30, hour = 23, tzinfo=timezone.utc)]
    },
    {
        "Station code" : 209,
        "pollutant name" : "NO2",
        "period" : [datetime(year = 2023, month= 9, day = 1, hour = 0, tzinfo=timezone.utc), datetime(year = 2023, month= 9, day = 30, hour = 23, tzinfo=timezone.utc)]
    },
    {
        "Station code" : 223,
        "pollutant name" : "O3",
        "period" : [datetime(year = 2023, month= 7, day = 1, hour = 0), datetime(year = 2023, month= 7, day = 31, hour = 23)]
    },
    {
        "Station code" : 224,
        "pollutant name" : "CO",
        "period" : [datetime(year = 2023, month= 10, day = 1, hour = 0), datetime(year = 2023, month= 10, day = 31, hour = 23)]
    },
    {
        "Station code" : 226,
        "pollutant name" : "PM10",
        "period" : [datetime(year = 2023, month= 8, day = 1, hour = 0), datetime(year = 2023, month= 8, day = 31, hour = 23)]
    },
    {
        "Station code" : 227,
        "pollutant name" : "PM2.5",
        "period" : [datetime(year = 2023, month= 12, day = 1, hour = 0), datetime(year = 2023, month= 12, day = 31, hour = 23)]
    }
]

output = {"target": {}}
cont = 0

for d in future:
    print(cont)
    station_code = str(d["Station code"])
    pollutant_name = d["pollutant name"]
    start_date, end_date = d["period"]
    
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train, X_val, y_train, y_val, scaler_y = obtain_train_data(pollutant, measurement, instrument, int(station_code), pollutant_name, scaler_x, scaler_y)
    X_test, date_range = obtain_test_data(start_date, end_date)
    preds = train_boosting_model(X_train, X_val, y_train, y_val, X_test, scaler_y)
    
    output["target"][station_code] = {str(date): float(pred) for date, pred in zip(date_range, preds)}
    cont += 1

with open("predictions/predictions_task_2.json", "w") as file:
    json.dump(output, file, indent = 4)