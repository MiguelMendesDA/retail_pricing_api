import json
import pickle
import datetime
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from peewee import (
    SqliteDatabase, Model, DateField, FloatField,
    IntegerField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from sklearn.base import BaseEstimator, TransformerMixin
import os
from playhouse.db_url import connect

# ---------------------- Database Setup ----------------------

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class PricePrediction(Model):
    sku = IntegerField()
    time_key = DateField()
    pvp_is_competitorA = FloatField()
    pvp_is_competitorB = FloatField()
    pvp_is_competitorA_actual = FloatField(null=True)
    pvp_is_competitorB_actual = FloatField(null=True)

    class Meta:
        database = DB
        indexes = ((('sku', 'time_key'), True),)

DB.connect()
DB.create_tables([PricePrediction], safe=True)


# ---------------------- Load Pipeline and Historical Data ----------------------

class TimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, time_column='time_key'):
        self.time_column = time_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.time_column] = pd.to_datetime(X[self.time_column])

        X['month'] = X[self.time_column].dt.month
        X['day'] = X[self.time_column].dt.day
        X['dayofweek'] = X[self.time_column].dt.dayofweek

        X['is_christmas_season'] = X['month'].isin([12]).astype(int)
        X['is_new_year'] = X[self.time_column].apply(
            lambda x: (x.month == 12 and x.day >= 26) or (x.month == 1 and x.day <= 5)
        ).astype(int)
        X['is_summer'] = X['month'].isin([6, 7, 8]).astype(int)
        X['is_back_to_school'] = (X['month'] == 9).astype(int)
        X['is_black_friday'] = X[self.time_column].apply(
            lambda x: x.month == 11 and x.weekday() == 4 and 23 <= x.day <= 29
        ).astype(int)

        return X


def load_pickle_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing pickle file: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)

model_A = load_pickle_safe("models/price_pipeline_A.pkl")
model_B = load_pickle_safe("models/price_pipeline_B.pkl")
hist_data_A = load_pickle_safe("models/historical_features_A.pkl")
hist_data_B = load_pickle_safe("models/historical_features_B.pkl")


# ---------------------- Helper Function for Prediction ----------------------

def predict_price(pipeline, hist_data, sku, target_date, competitor='A'):
    if sku not in hist_data:
        raise ValueError(f'SKU {sku} not found.')

    df = hist_data[sku].copy()
    df['time_key'] = pd.to_datetime(df['time_key'])

    target_date = pd.Timestamp(target_date)

    same_day = df[
        (df['time_key'].dt.day == target_date.day) &
        (df['time_key'].dt.month == target_date.month) &
        (df['time_key'] < target_date)
    ]

    if not same_day.empty:
        ref_row = same_day.sort_values('time_key').iloc[-1]
    else:
        df_filtered = df[df['time_key'] < target_date]
        if df_filtered.empty:
            raise ValueError(f'Insufficient history for SKU {sku} before {target_date}.')
        ref_row = df_filtered.sort_values('time_key').iloc[-1]

    if competitor == 'A':
        input_data = {
            'time_key': target_date,
            'lag_diffA': ref_row['lag_diffA'],
            'lag_diffA_sl4': ref_row['lag_diffA_sl4'],
            'diffA_std_sku': ref_row['diffA_std_sku'],
            'chain_price': ref_row['chain_price']
        }
    elif competitor == 'B':
        input_data = {
            'time_key': target_date,
            'lag_diffB': ref_row['lag_diffB'],
            'lag_diffB_sl4': ref_row['lag_diffB_sl4'],
            'diffB_std_sku': ref_row['diffB_std_sku'],
            'chain_price': ref_row['chain_price']
        }
    else:
        raise ValueError("Competitor must be 'A' or 'B'")

    input_df = pd.DataFrame([input_data])
    return float(pipeline.predict(input_df)[0])


# ---------------------- Validation ----------------------

def validate_time_key_int(time_key_int):
    try:
        time_str = str(time_key_int)
        date_obj = datetime.datetime.strptime(time_str, "%Y%m%d").date()
        return True, date_obj
    except ValueError:
        return False, "Invalid time_key format. Expected integer in YYYYMMDD format."


def check_forecast_input(payload):
    if "sku" not in payload:
        return False, "Missing field: sku"

    try:
        payload["sku"] = int(payload["sku"])  # normalize to int
    except (ValueError, TypeError):
        return False, "Invalid sku: must be numeric"

    if "time_key" not in payload:
        return False, "Missing field: time_key"
    
    is_valid, result = validate_time_key_int(payload["time_key"])
    if not is_valid:
        return False, result

    payload["time_key"] = result
    return True, ""


def check_actual_input(payload):
    required = ["sku", "time_key", "pvp_is_competitorA_actual", "pvp_is_competitorB_actual"]
    for field in required:
        if field not in payload:
            return False, f"Missing field: {field}"
    
    try:
        payload["sku"] = int(payload["sku"])  # normalize to int
    except (ValueError, TypeError):
        return False, "Invalid sku: must be numeric"

    is_valid, result = validate_time_key_int(payload["time_key"])
    if not is_valid:
        return False, result
    payload["time_key"] = result

    try:
        float(payload["pvp_is_competitorA_actual"])
        float(payload["pvp_is_competitorB_actual"])
    except ValueError:
        return False, "Actual prices must be numeric"

    return True, ""


# ---------------------- Flask App ----------------------

app = Flask(__name__)
CORS(app)

@app.route("/forecast_prices/", methods=["POST"])
def forecast_prices():
    payload = request.get_json()
    is_valid, error = check_forecast_input(payload)
    if not is_valid:
        return jsonify({"error": error}), 422

    sku = payload["sku"]
    time_key = payload["time_key"]

    try:
        PricePrediction.get(
            (PricePrediction.sku == sku) &
            (PricePrediction.time_key == time_key)
        )
        return jsonify({"error": "Forecast already exists for this sku and time_key"}), 422
    except PricePrediction.DoesNotExist:
        pass

    try:
        price_A = predict_price(model_A, hist_data_A, sku, time_key, competitor='A')
        price_B = predict_price(model_B, hist_data_B, sku, time_key, competitor='B')

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    try:
        record = PricePrediction.create(
            sku=sku,
            time_key=time_key,
            pvp_is_competitorA=price_A,
            pvp_is_competitorB=price_B,
        )
    except IntegrityError:
        return jsonify({"error": "Forecast already exists for this sku and time_key"}), 422

    return jsonify({
        "sku": sku,
        "time_key": int(time_key.strftime("%Y%m%d")),
        "pvp_is_competitorA": price_A,
        "pvp_is_competitorB": price_B
    })


@app.route("/actual_prices/", methods=["POST"])
def actual_prices():
    payload = request.get_json()
    is_valid, error = check_actual_input(payload)
    if not is_valid:
        return jsonify({"error": error}), 422

    sku = payload["sku"]
    time_key = payload["time_key"]

    try:
        record = PricePrediction.get((PricePrediction.sku == sku) & (PricePrediction.time_key == time_key))
    except PricePrediction.DoesNotExist:
        return jsonify({"error": "No forecast exists for this sku and time_key"}), 422

    record.pvp_is_competitorA_actual = float(payload["pvp_is_competitorA_actual"])
    record.pvp_is_competitorB_actual = float(payload["pvp_is_competitorB_actual"])
    record.save()

    return jsonify({
        "sku": record.sku,
        "time_key": int(record.time_key.strftime("%Y%m%d")),
        "pvp_is_competitorA": record.pvp_is_competitorA,
        "pvp_is_competitorB": record.pvp_is_competitorB,
        "pvp_is_competitorA_actual": record.pvp_is_competitorA_actual,
        "pvp_is_competitorB_actual": record.pvp_is_competitorB_actual,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
