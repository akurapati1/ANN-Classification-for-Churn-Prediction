"""Feature engineering shared by the Streamlit app and tests (no TensorFlow dependency)."""
import pickle
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent


def load_preprocessors(base_dir: Path = BASE_DIR):
    """Load the fitted gender encoder, geography one-hot encoder and scaler saved by experiments.ipynb."""
    def _load(name):
        with open(base_dir / name, "rb") as fh:
            return pickle.load(fh)

    return _load("label_encoder_gender.pkl"), _load("one_hot_encoder_geo.pkl"), _load("scaler.pkl")


def build_features(customer: dict, gender_encoder, geo_encoder, scaler):
    """Turn one raw customer record into the scaled feature matrix the model was trained on.

    Column order must match training: numeric/encoded columns first, then the one-hot geography columns.
    """
    row = pd.DataFrame(
        {
            "CreditScore": [customer["credit_score"]],
            "Gender": [gender_encoder.transform([customer["gender"]])[0]],
            "Age": [customer["age"]],
            "Tenure": [customer["tenure"]],
            "Balance": [customer["balance"]],
            "NumOfProducts": [customer["num_of_products"]],
            "HasCrCard": [customer["has_cr_card"]],
            "IsActiveMember": [customer["is_active_member"]],
            "EstimatedSalary": [customer["estimated_salary"]],
        }
    )
    geo = pd.DataFrame(
        geo_encoder.transform([[customer["geography"]]]).toarray(),
        columns=geo_encoder.get_feature_names_out(["Geography"]),
    )
    return scaler.transform(pd.concat([row, geo], axis=1))
