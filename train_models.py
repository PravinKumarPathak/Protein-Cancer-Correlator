# train_models.py
import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

# Create required directories
os.makedirs("models", exist_ok=True)
os.makedirs("preprocessors", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

# Load dataset
df = pd.read_csv('Cancer_dataset.csv')
drop_cols = ["Entry", "Entry Name", "Protein names", "Gene Names", "Sequence", "selected", "shared name"]
df_clean = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

X = df_clean.drop(columns=['target'])
y = df_clean['target']

# Preprocessing
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)

# Save preprocessors
joblib.dump(imputer, 'preprocessors/imputer.pkl')
joblib.dump(scaler, 'preprocessors/scaler.pkl')

# === Train ML Model ===
ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
ml_model.fit(X_train, y_train)
ml_preds = ml_model.predict(X_test)

ml_metrics = {
    "accuracy": accuracy_score(y_test, ml_preds),
    "f1_score": f1_score(y_test, ml_preds, average='weighted'),
    "roc_auc": roc_auc_score(y_test, ml_preds),
    "precision": precision_score(y_test, ml_preds),
    "recall": recall_score(y_test, ml_preds),
    "confusion_matrix": confusion_matrix(y_test, ml_preds).tolist()
}

joblib.dump(ml_model, 'models/ml_model.pkl')

# === Train DL Model ===
class_weights = dict(zip(*np.unique(y_train, return_counts=True)))
for k in class_weights:
    class_weights[k] = len(y_train) / (2.0 * class_weights[k])

dl_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

dl_model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
dl_model.fit(X_train, y_train, validation_data=(X_test, y_test),
             epochs=100, batch_size=64,
             class_weight=class_weights,
             callbacks=[EarlyStopping(patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(patience=5, factor=0.5)])

dl_model.save('models/dl_model.h5')
dl_preds = (dl_model.predict(X_test) > 0.5).astype(int)

# DL metrics
loss, dl_accuracy = dl_model.evaluate(X_test, y_test)
dl_metrics = {
    "accuracy": dl_accuracy,
    "f1_score": f1_score(y_test, dl_preds, average='weighted'),
    "roc_auc": roc_auc_score(y_test, dl_preds),
    "precision": precision_score(y_test, dl_preds),
    "recall": recall_score(y_test, dl_preds),
    "confusion_matrix": confusion_matrix(y_test, dl_preds).tolist()
}

# Save all metrics
with open("metrics/metrics.json", "w") as f:
    json.dump({"ml": ml_metrics, "dl": dl_metrics}, f, indent=4)
