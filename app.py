# app.py
import os
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from flask import Flask, request, render_template

# Load pre-trained models and tools
mlModel = joblib.load('ml_model.pkl')
dlModel = load_model('dl_model.h5')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')

df = pd.read_csv('dataset.csv')
drop_cols = ["Entry", "Entry Name", "Protein names", "Gene Names", "Sequence", "selected", "shared name"]

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        protein_Name = request.form['proteinInput']
        choice = request.form['model_choice']

        if protein_Name not in df['Protein names'].values:
            return "Error! Invalid input. Please enter a valid protein name."

        testdf = df[df['Protein names'] == protein_Name]
        testdf = testdf.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

        testX = testdf.drop(columns=['target'])

        testX_imputed = pd.DataFrame(imputer.transform(testX), columns=testX.columns)
        testX_scaled = pd.DataFrame(scaler.transform(testX_imputed), columns=testX.columns)

        if choice == 'ml':
            pred = mlModel.predict(testX_scaled)[0]
            res = 'This protein is related to cancer' if pred == 1 else 'This protein is not related to cancer'
            return render_template('result.html', result={'modType': 'ML', 'modName': 'Random Forest', 'pred': res})
        else:
            pred = dlModel.predict(testX_scaled)[0][0]
            pred_binary = int(pred > 0.5)
            res = 'This protein is related to cancer' if pred_binary == 1 else 'This protein is not related to cancer'
            return render_template('result.html', result={'modType': 'DL', 'modName': 'Neural Net', 'pred': res})

    return render_template('form.html')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
