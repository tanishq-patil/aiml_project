
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and preprocessing objects
bundle = joblib.load("aqi_model.pkl")

scaler = bundle["scaler"]
poly = bundle["poly"]
model = bundle["model"]
best_model = bundle.get("best_model")
features = bundle["features"]
ranges = bundle["ranges"]
units = {
    "PM2.5": "µg/m³",
    "PM10": "µg/m³",
    "NO": "µg/m³",
    "NO2": "µg/m³",
    "NOx": "µg/m³",
    "NH3": "µg/m³",
    "CO": "mg/m³",
    "SO2": "µg/m³",
    "O3": "µg/m³",
    "Benzene": "µg/m³",
    "Toluene": "µg/m³",
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
            if best_model is None:
                raise ValueError(
                    "The optimized model is missing from aqi_model.pkl. "
                    "Run the notebook's model export cell to update it."
                )

            for col in features:
                value = float(request.form[col])

                if value < 0:
                    raise ValueError(
                        f"{col} cannot be negative."
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

            # lr2 uses the scaler and polynomial features saved with it.
            scaled_data = scaler.transform(user_data)
            polynomial_data = poly.transform(scaled_data)

            predictions = [
                {
                    "name": "Polynomial model",
                    "aqi": float(model.predict(polynomial_data)[0])
                },
                {
                    "name": "Optimized Polyomial model",
                    "aqi": float(best_model.predict(user_data)[0])
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
        units=units,
        values=values,
        predictions=predictions,
        error=error,
        warnings=warnings
    )


if __name__ == "__main__":
    app.run(debug=True)
