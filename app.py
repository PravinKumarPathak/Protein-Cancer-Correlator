# app.py
import joblib
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

# Load artifacts
df = pd.read_csv("dataset.csv")
drop_cols = ["Entry", "Entry Name", "Protein names", "Gene Names", "Sequence", "selected", "shared name"]

ml_model = joblib.load("models/ml_model.pkl")
dl_model = load_model("models/dl_model.h5")
imputer = joblib.load("preprocessors/imputer.pkl")
scaler = joblib.load("preprocessors/scaler.pkl")

with open("metrics/metrics.json", "r") as f:
    metrics = json.load(f)


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def hello():
    if request.method == "POST":
        protein = request.form["proteinInput"]
        choice = request.form["model_choice"]

        if protein not in df["Protein names"].values:
            return "Error!! Invalid input. Please enter a valid protein name."

        test_df = df[df["Protein names"] == protein]
        test_df = test_df.drop(columns=[col for col in drop_cols if col in test_df.columns], errors='ignore')
        test_X = test_df.drop(columns=['target'])
        test_X = pd.DataFrame(scaler.transform(imputer.transform(test_X)), columns=test_X.columns)

        if choice == "ml":
            pred = ml_model.predict(test_X)[0]
            res = 'This protein is related to cancer' if pred == 1 else 'This protein is not related to cancer'
            model_info = {
                "model": "Random Forest Classifier",
                "type": "Machine Learning",
                "accuracy": f"{metrics['ml']['accuracy']*100:.2f}%",
                "f1_score": f"{metrics['ml']['f1_score']*100:.2f}%",
                "roc_auc": f"{metrics['ml']['roc_auc']*100:.2f}%"
            }
        else:
            pred = (dl_model.predict(test_X)[0][0] > 0.5)
            res = 'This protein is related to cancer' if pred == 1 else 'This protein is not related to cancer'
            model_info = {
                "model": "Artificial Neural Network",
                "type": "Deep Learning",
                "accuracy": f"{metrics['dl']['accuracy']*100:.2f}%",
                "f1_score": f"{metrics['dl']['f1_score']*100:.2f}%",
                "roc_auc": f"{metrics['dl']['roc_auc']*100:.2f}%"
            }

        return render_template("result.html", result=res, model_info=model_info)

    return render_template("form.html")


if __name__ == '__main__':
    app.run(debug=False)
