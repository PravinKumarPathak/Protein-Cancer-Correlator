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

# Load full metrics including precision, recall, and confusion matrix
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
            responsibility = "YES" if pred == 1 else "NO"
            performance = metrics["ml"]
            model_used = "Random Forest Classifier"
        else:
            pred = (dl_model.predict(test_X)[0][0] > 0.5)
            responsibility = "YES" if pred == 1 else "NO"
            performance = metrics["dl"]
            model_used = "Artificial Neural Network"

        message = f'The protein {protein} is {"associated" if responsibility == "YES" else "not associated"} with cancer.'

        # Confusion matrix is a 2x2 list: [[TN, FP], [FN, TP]]
        cm_matrix = performance["confusion_matrix"]
        confusion_matrix = {
            "TP": cm_matrix[1][1],
            "FN": cm_matrix[1][0],
            "FP": cm_matrix[0][1],
            "TN": cm_matrix[0][0],
        }

        return render_template(
            "result.html",
            protein_name=protein,
            responsibility=responsibility,
            result=message,
            model_used=model_used,
            accuracy=f"{performance['accuracy']*100:.2f}%",
            precision=f"{performance['precision']*100:.2f}%",
            recall=f"{performance['recall']*100:.2f}%",
            f1_score=f"{performance['f1_score']*100:.2f}%",
            cm=confusion_matrix
        )

    return render_template("form.html")

# Route for About page
@app.route("/about")
def about():
    return render_template("about.html")



if __name__ == '__main__':
    app.run(debug=False)
