# Importing Modules
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from flask import Flask, render_template, request


# Data Pre-Processing
## Read dataset
df = pd.read_csv('dataset.csv')

## Drop irrelevant columns
drop_cols = ["Entry", "Entry Name", "Protein names", "Gene Names", "Sequence", "selected", "shared name"]
dframe = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

## Separate features and target
y = dframe["target"]
X = dframe.drop(columns=["target"])

## Impute missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

## Feature Scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

## Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

## Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,
    random_state=42,
    stratify=y_resampled
)


# ML Model
mlModel = RandomForestClassifier(n_estimators=100, random_state=42)
mlModel.fit(X_train, y_train)

ml_pred = mlModel.predict(X_test)
ml_accuracy = accuracy_score(y_test, ml_pred)
ml_f1 = f1_score(y_test, ml_pred, average='weighted')
ml_roc_auc = roc_auc_score(y_test, ml_pred)


# DL Model
## Class weights
classes = np.unique(y_train)
weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

## Model Architecture
n_cols = X_train.shape[1]
dlModel = Sequential()
dlModel.add(Dense(128, activation='relu', input_shape=(n_cols,), kernel_regularizer=l2(0.001)))
dlModel.add(BatchNormalization())
dlModel.add(Dropout(0.4))
dlModel.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
dlModel.add(BatchNormalization())
dlModel.add(Dropout(0.3))
dlModel.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
dlModel.add(BatchNormalization())
dlModel.add(Dropout(0.2))
dlModel.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=0.0005)
dlModel.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

dlModel.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

loss, dl_accuracy = dlModel.evaluate(X_test, y_test)

dl_pred_prob = dlModel.predict(X_test)
dl_pred = (dl_pred_prob > 0.5).astype(int)

dl_f1 = f1_score(y_test, dl_pred, average='weighted')
dl_roc_auc = roc_auc_score(y_test, dl_pred)


# Flask Application
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
        testY = testdf['target']

        testX_imputed = pd.DataFrame(imputer.transform(testX), columns=testX.columns)
        testX_scaled = pd.DataFrame(scaler.transform(testX_imputed), columns=testX.columns)

        test_mlPred = mlModel.predict(testX_scaled)
        test_dlPred_prob = dlModel.predict(testX_scaled)
        test_dlPred = (test_dlPred_prob > 0.5).astype(int)

        if choice == 'ml':
            predtn = test_mlPred[0]
            modType = 'Machine Learning'
            modName = 'Random Forest Classifier'
            acu = ml_accuracy
            fsc = ml_f1
            ras = ml_roc_auc
        else:
            predtn = test_dlPred[0][0]
            modType = 'Deep Learning'
            modName = 'Artificial Neural Network'
            acu = dl_accuracy
            fsc = dl_f1
            ras = dl_roc_auc

        res = 'This protein is related to cancer' if predtn == 1 else 'This protein is not related to cancer'

        result = {
            'modType': modType,
            'modName': modName,
            'pred': res,
            'acu': f'{acu * 100:.2f}%',
            'fsc': f'{fsc * 100:.2f}%',
            'ras': f'{ras * 100:.2f}%'
        }
        return render_template('result.html', result=result)

    return render_template('form.html')


if __name__ == '__main__':
    app.run(debug=False)

