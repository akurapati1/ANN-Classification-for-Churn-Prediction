"""Streamlit UI for the bank-customer churn ANN."""
import streamlit as st
from tensorflow.keras.models import load_model

from churn_features import BASE_DIR, build_features, load_preprocessors

THRESHOLD = 0.5


@st.cache_resource
def load_artifacts():
    model = load_model(BASE_DIR / "model.h5")
    return (model, *load_preprocessors())


model, gender_encoder, geo_encoder, scaler = load_artifacts()

st.title("Customer Churn Prediction")
st.caption("A neural network trained on the Churn_Modelling dataset (~86% validation accuracy).")

col1, col2 = st.columns(2)
with col1:
    geography = st.selectbox("Geography", geo_encoder.categories_[0])
    gender = st.selectbox("Gender", gender_encoder.classes_)
    age = st.slider("Age", 18, 100, 40)
    tenure = st.slider("Tenure (years)", 0, 10, 5)
    num_of_products = st.slider("Number of products", 1, 4, 1)
with col2:
    credit_score = st.number_input("Credit score", min_value=300, max_value=900, value=650)
    balance = st.number_input("Balance", min_value=0.0, value=0.0, step=1000.0)
    estimated_salary = st.number_input("Estimated salary", min_value=0.0, value=50000.0, step=1000.0)
    has_cr_card = st.selectbox("Has credit card", [0, 1], format_func=lambda v: "Yes" if v else "No")
    is_active_member = st.selectbox("Is active member", [0, 1], format_func=lambda v: "Yes" if v else "No")

features = build_features(
    {
        "geography": geography, "gender": gender, "age": age, "tenure": tenure, "balance": balance,
        "credit_score": credit_score, "estimated_salary": estimated_salary, "num_of_products": num_of_products,
        "has_cr_card": has_cr_card, "is_active_member": is_active_member,
    },
    gender_encoder, geo_encoder, scaler,
)
probability = float(model.predict(features, verbose=0)[0][0])

st.subheader("Prediction")
st.progress(min(max(probability, 0.0), 1.0))
if probability > THRESHOLD:
    st.error("The customer is likely to exit the bank.")
else:
    st.success("The customer is more likely to stay with the bank.")
st.write(f"Exit probability: **{probability:.2%}**")
