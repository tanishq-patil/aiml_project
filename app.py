
import math

import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)

# Load the trained model and preprocessing objects
bundle = joblib.load("aqi_model.pkl")

scaler = bundle["scaler"]
poly = bundle["poly"]
model = bundle["model"]
best_model = bundle.get("best_model")
model_features = list(bundle["features"])
features = [
    "pm2_5",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
    "dust",
    "temperature",
    "humidity",
    "wind_speed",
    "rain",
]
ranges = {
    "pm2_5": {"min": 0.4, "max": 111.3},
    "pm10": {"min": 0.6, "max": 141.1},
    "no2": {"min": 0.0, "max": 149.3},
    "so2": {"min": 0.3, "max": 55.7},
    "co": {"min": 59.0, "max": 4146.0},
    "o3": {"min": 0.0, "max": 232.0},
    "dust": {"min": 0.0, "max": 117.0},
    "temperature": {"min": 10.5, "max": 40.0},
    "humidity": {"min": 9.0, "max": 100.0},
    "wind_speed": {"min": 0.0, "max": 40.0},
    "rain": {"min": 0.0, "max": 20.6},
}
input_limits = {
    "pm2_5": {"min": 0.0, "max": 1000.0},
    "pm10": {"min": 0.0, "max": 1000.0},
    "no2": {"min": 0.0, "max": 1000.0},
    "so2": {"min": 0.0, "max": 1000.0},
    "co": {"min": 0.0, "max": 100000.0},
    "o3": {"min": 0.0, "max": 1000.0},
    "dust": {"min": 0.0, "max": 1000.0},
    "temperature": {"min": -50.0, "max": 60.0},
    "humidity": {"min": 0.0, "max": 100.0},
    "wind_speed": {"min": 0.0, "max": 200.0},
    "rain": {"min": 0.0, "max": 1000.0},
}
labels = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "no2": "NO2",
    "so2": "SO2",
    "co": "CO",
    "o3": "O3",
    "dust": "Dust",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "wind_speed": "Wind speed",
    "rain": "Rain",
}
units = {
    "pm2_5": "µg/m³",
    "pm10": "µg/m³",
    "no2": "µg/m³",
    "so2": "µg/m³",
    "co": "µg/m³",
    "o3": "µg/m³",
    "dust": "µg/m³",
    "temperature": "°C",
    "humidity": "%",
    "wind_speed": "dataset units",
    "rain": "dataset units",
}


def get_aqi_category(aqi):
    if aqi < 0 or aqi > 500:
        return "Outside standard AQI scale"
    elif aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


@app.route("/", methods=["GET", "POST"])
def home():
    predictions = []
    error = None
    warnings = []
    values = {}

    if request.method == "POST":
        try:
            if model_features != features:
                raise ValueError(
                    "The saved model uses a different set of input attributes. "
                    "Run the updated training and export cells in aqi.ipynb "
                    "to regenerate aqi_model.pkl."
                )

            if best_model is None:
                raise ValueError(
                    "The optimized model is missing from aqi_model.pkl. "
                    "Run the notebook's model export cell to update it."
                )

            for col in features:
                value = float(request.form[col])

                if not math.isfinite(value):
                    raise ValueError(
                        f"{labels[col]} must be a finite number."
                    )

                minimum = input_limits[col]["min"]
                maximum = input_limits[col]["max"]
                if value < minimum or value > maximum:
                    raise ValueError(
                        f"{labels[col]} must be between "
                        f"{minimum:g} and {maximum:g} {units[col]}."
                    )

                values[col] = value

                low = ranges[col]["min"]
                high = ranges[col]["max"]

                if value < low or value > high:
                    warnings.append(
                        f"{col} is outside the training "
                        f"range ({low:.2f}–{high:.2f})."
                    )

            # Keep the exact feature order used during training
            user_data = pd.DataFrame(
                [[values[col] for col in features]],
                columns=features
            )

            scaled_data = scaler.transform(user_data)
            degree_three_data = PolynomialFeatures(degree=3).fit_transform(
                scaled_data
            )
            degree_four_data = poly.transform(scaled_data)

            predictions = [
                {
                    "name": "Polynomial regression (degree 3)",
                    "aqi": float(model.predict(degree_three_data)[0])
                },
                {
                    "name": "Polynomial regression (degree 4)",
                    "aqi": float(best_model.predict(degree_four_data)[0])
                }
            ]
            for prediction in predictions:
                prediction["category"] = get_aqi_category(
                    prediction["aqi"]
                )

        except (ValueError, TypeError) as e:
            error = str(e) or "Please enter valid numeric values."

    return render_template(
        "index.html",
        features=features,
        ranges=ranges,
        input_limits=input_limits,
        labels=labels,
        units=units,
        values=values,
        predictions=predictions,
        error=error,
        warnings=warnings
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
