# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import pickle
# import os

# # ── Paths ────────────────────────────────────────────────────────
# BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH  = os.path.join(BASE_DIR, '..', 'data', 'diabetes.csv')
# MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

# # ── Feature columns (same order the model was trained on) ────────
# FEATURES = [
#     'Pregnancies', 'Glucose', 'BloodPressure',
#     'SkinThickness', 'Insulin', 'BMI',
#     'DiabetesPedigreeFunction', 'Age'
# ]
# TARGET = 'Outcome'


# # ────────────────────────────────────────────────────────────────
# # 1. Preprocessing
# # ────────────────────────────────────────────────────────────────
# def preprocess(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Replace biologically impossible zero values with column median,
#     then return a clean copy.  (Zeros in Glucose, BP, BMI etc.
#     are missing-data placeholders in the Pima dataset.)
#     """
#     df = df.copy()
#     zero_not_allowed = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
#     for col in zero_not_allowed:
#         if col in df.columns:
#             df[col] = df[col].replace(0, np.nan)
#             df[col].fillna(df[col].median(), inplace=True)
#     return df


# # ────────────────────────────────────────────────────────────────
# # 2. Training  —  run this once to create model.pkl
# # ────────────────────────────────────────────────────────────────
# def train_and_save(data_path: str = DATA_PATH,
#                    model_path: str = MODEL_PATH) -> dict:
#     """
#     Load the CSV, preprocess, train a Random Forest, save model + scaler.
#     Returns a dict with accuracy and classification report.
#     """
#     df = pd.read_csv(data_path)
#     df = preprocess(df)

#     X = df[FEATURES]
#     y = df[TARGET]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )

#     # Scale features
#     scaler = StandardScaler()
#     X_train_sc = scaler.fit_transform(X_train)
#     X_test_sc  = scaler.transform(X_test)

#     # Model
#     model = RandomForestClassifier(
#         n_estimators=100,
#         max_depth=6,
#         random_state=42,
#         class_weight='balanced'   # handles slight class imbalance
#     )
#     model.fit(X_train_sc, y_train)

#     # Evaluate
#     y_pred = model.predict(X_test_sc)
#     acc    = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred,
#                                    target_names=['No Diabetes', 'Diabetes'])
#     cm     = confusion_matrix(y_test, y_pred).tolist()

#     # Save both model AND scaler together so predict() always uses the same scaling
#     artifact = {'model': model, 'scaler': scaler, 'features': FEATURES}
#     with open(model_path, 'wb') as f:
#         pickle.dump(artifact, f)

#     print(f"[train] Accuracy: {acc:.4f}")
#     print(report)
#     print(f"[train] Saved → {model_path}")

#     return {'accuracy': round(acc, 4), 'report': report, 'confusion_matrix': cm}


# # ────────────────────────────────────────────────────────────────
# # 3. Loading
# # ────────────────────────────────────────────────────────────────
# _artifact = None   # module-level cache so we load only once

# def load_model(model_path: str = MODEL_PATH) -> dict:
#     """Load model.pkl and cache it in memory."""
#     global _artifact
#     if _artifact is None:
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(
#                 f"model.pkl not found at {model_path}. "
#                 "Run `python model.py` (or train.py) first."
#             )
#         with open(model_path, 'rb') as f:
#             _artifact = pickle.load(f)
#     return _artifact


# # ────────────────────────────────────────────────────────────────
# # 4. Prediction
# # ────────────────────────────────────────────────────────────────
# def predict(input_data: dict) -> dict:
#     """
#     Make a prediction from a dict of raw patient values.

#     Expected keys (matching frontend field names):
#         pregnancies, glucose, bp, skin, insulin, bmi, dpf, age

#     Returns:
#         {
#           'probability': 73,          # int 0-100
#           'outcome': 1,               # 0 = no diabetes, 1 = diabetes
#           'risk_level': 'High',       # 'Low' | 'Moderate' | 'High'
#           'feature_importances': {...} # top contributing features
#         }
#     """
#     artifact = load_model()
#     model    = artifact['model']
#     scaler   = artifact['scaler']

#     # Map frontend keys → dataset column names
#     row = {
#         'Pregnancies':             float(input_data.get('pregnancies', 0)),
#         'Glucose':                 float(input_data.get('glucose', 0)),
#         'BloodPressure':           float(input_data.get('bp', 0)),
#         'SkinThickness':           float(input_data.get('skin', 0)),
#         'Insulin':                 float(input_data.get('insulin', 0)),
#         'BMI':                     float(input_data.get('bmi', 0)),
#         'DiabetesPedigreeFunction':float(input_data.get('dpf', 0)),
#         'Age':                     float(input_data.get('age', 0)),
#     }

#     # Build DataFrame in training column order
#     X = pd.DataFrame([row])[FEATURES]

#     # Apply the SAME preprocessing used during training
#     X = preprocess(X)

#     # Scale
#     X_sc = scaler.transform(X)

#     # Predict
#     prob    = float(model.predict_proba(X_sc)[0][1])
#     outcome = int(prob >= 0.5)
#     prob_pct = round(prob * 100)

#     # Risk level thresholds
#     if prob_pct < 30:
#         risk = 'Low'
#     elif prob_pct < 60:
#         risk = 'Moderate'
#     else:
#         risk = 'High'

#     # Feature importances mapped back to readable names
#     readable = {
#         'Pregnancies': 'Pregnancies',
#         'Glucose': 'Glucose level',
#         'BloodPressure': 'Blood pressure',
#         'SkinThickness': 'Skin thickness',
#         'Insulin': 'Insulin level',
#         'BMI': 'BMI',
#         'DiabetesPedigreeFunction': 'Diabetes pedigree',
#         'Age': 'Age',
#     }
#     importances = {
#         readable[feat]: round(float(imp), 4)
#         for feat, imp in zip(FEATURES, model.feature_importances_)
#     }
#     top_factors = sorted(importances, key=importances.get, reverse=True)[:3]

#     return {
#         'probability':        prob_pct,
#         'outcome':            outcome,
#         'risk_level':         risk,
#         'feature_importances':importances,
#         'top_factors':        top_factors,
#     }


# # ────────────────────────────────────────────────────────────────
# # 5. Run directly to train
# # ────────────────────────────────────────────────────────────────
# if __name__ == '__main__':
#     results = train_and_save()
#     print("\nTop feature importances (after training):")
#     art = load_model()
#     for feat, imp in sorted(
#         zip(FEATURES, art['model'].feature_importances_),
#         key=lambda x: x[1], reverse=True
#     ):
#         bar = '█' * int(imp * 40)
#         print(f"  {feat:<28} {bar}  {imp:.4f}")


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, '..', 'data', 'diabetes.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

# ── Feature columns ───────────────────────────────────────────────
FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure',
    'SkinThickness', 'Insulin', 'BMI',
    'DiabetesPedigreeFunction', 'Age'
]
TARGET = 'Outcome'


# ────────────────────────────────────────────────────────────────
# 1. Preprocessing
# ────────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    zero_not_allowed = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    for col in zero_not_allowed:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())

    return df


# ────────────────────────────────────────────────────────────────
# 2. Training
# ────────────────────────────────────────────────────────────────
def train_and_save(data_path=DATA_PATH, model_path=MODEL_PATH):
    df = pd.read_csv(data_path)
    df = preprocess(df)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        class_weight='balanced'
    )

    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)

    print(f"[train] Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # ✅ ALWAYS save as dict
    artifact = {
        'model': model,
        'scaler': scaler,
        'features': FEATURES
    }

    with open(model_path, 'wb') as f:
        pickle.dump(artifact, f)

    print(f"[train] Model saved at: {model_path}")


# ────────────────────────────────────────────────────────────────
# 3. Load model
# ────────────────────────────────────────────────────────────────
_artifact = None

def load_model(model_path=MODEL_PATH):
    global _artifact

    if _artifact is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError("model.pkl not found. Run training first.")

        with open(model_path, 'rb') as f:
            _artifact = pickle.load(f)

        # ✅ Safety check
        if not isinstance(_artifact, dict):
            raise ValueError("Invalid model file. Delete model.pkl and retrain.")

    return _artifact


# ────────────────────────────────────────────────────────────────
# 4. Prediction
# ────────────────────────────────────────────────────────────────
def predict(input_data: dict):

    # ✅ Safety check
    if not isinstance(input_data, dict):
        raise ValueError("Input must be a dictionary")

    artifact = load_model()
    model  = artifact['model']
    scaler = artifact['scaler']

    # Map input
    row = {
        'Pregnancies': float(input_data.get('pregnancies', 0)),
        'Glucose': float(input_data.get('glucose', 0)),
        'BloodPressure': float(input_data.get('bp', 0)),
        'SkinThickness': float(input_data.get('skin', 0)),
        'Insulin': float(input_data.get('insulin', 0)),
        'BMI': float(input_data.get('bmi', 0)),
        'DiabetesPedigreeFunction': float(input_data.get('dpf', 0)),
        'Age': float(input_data.get('age', 0)),
    }

    # Convert to DataFrame
    X = pd.DataFrame([row])[FEATURES]

    # Preprocess
    X = preprocess(X)

    # Scale
    X_sc = scaler.transform(X)

    # Predict
    prob = float(model.predict_proba(X_sc)[0][1])
    prob_pct = round(prob * 100)
    outcome = int(prob >= 0.5)

    # Risk level
    if prob_pct < 30:
        risk = 'Low'
    elif prob_pct < 60:
        risk = 'Moderate'
    else:
        risk = 'High'

    # Feature importance
    readable = {
        'Pregnancies': 'Pregnancies',
        'Glucose': 'Glucose level',
        'BloodPressure': 'Blood pressure',
        'SkinThickness': 'Skin thickness',
        'Insulin': 'Insulin level',
        'BMI': 'BMI',
        'DiabetesPedigreeFunction': 'Diabetes pedigree',
        'Age': 'Age',
    }

    importances = {
        readable[f]: round(float(i), 4)
        for f, i in zip(FEATURES, model.feature_importances_)
    }

    top_factors = sorted(importances, key=importances.get, reverse=True)[:3]

    # Optional analysis text
    analysis = f"Based on the provided clinical values, the estimated diabetes probability is {prob_pct}%. Key influencing factors include {', '.join(top_factors)}."

    recommendation = None
    if risk == 'High':
        recommendation = "Consult a doctor immediately and consider blood glucose testing."
    elif risk == 'Moderate':
        recommendation = "Maintain a healthy diet and monitor glucose regularly."
    else:
        recommendation = "Keep maintaining a healthy lifestyle."

    return {
        'probability': prob_pct,
        'outcome': outcome,
        'risk_level': risk,
        'feature_importances': importances,
        'top_factors': top_factors,
        'analysis': analysis,
        'recommendation': recommendation
    }


# ────────────────────────────────────────────────────────────────
# 5. Run training
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_and_save()