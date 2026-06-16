from flask import Flask, render_template, request
import joblib
import pandas as pd
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "..", "templates")
)

model_path = os.path.join(BASE_DIR, "model.pkl")
model = joblib.load(model_path)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_json = request.form["json_data"]

        data = json.loads(input_json)

        df = pd.DataFrame([data])

        prediction = model.predict(df)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Anomaly Label: {prediction}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    print("Current directory:", os.getcwd())
    print("Template folder:", app.template_folder)

    app.run(host="0.0.0.0", port=5000, debug=True)