import streamlit as st
import joblib
import numpy as np

# Load your trained model
model = joblib.load("churn_model.pkl")

st.title("Customer Churn Prediction")

st.markdown("Provide the following details to predict the churn status:")

# Example features – adjust based on your actual model input
region = st.selectbox("Region", ['DAKAR', 'THIES', 'DIOURBEL', 'FATICK'])  # example regions
tenure = st.selectbox("Tenure", ['0-3 month', '3-6 month', '6-9 month', '9-12 month', '12-15 month', '15-18 month', '18-21 month', '21-24 month', 'K > 24 month'])
montant = st.number_input("Montant", min_value=0.0, step=1.0)
frequence_rech = st.number_input("Fréquence Recharge", min_value=0.0, step=1.0)
revenue = st.number_input("Revenue", min_value=0.0, step=1.0)
arpu_segment = st.number_input("ARPU Segment", min_value=0.0, step=1.0)
frequence = st.number_input("Frequence", min_value=0.0, step=1.0)
data_volume = st.number_input("Data Volume", min_value=0.0, step=1.0)
on_net = st.number_input("On Net", min_value=0.0, step=1.0)
orange = st.number_input("Orange", min_value=0.0, step=1.0)
tigo = st.number_input("Tigo", min_value=0.0, step=1.0)
zone1 = st.number_input("Zone 1", min_value=0.0, step=1.0)
zone2 = st.number_input("Zone 2", min_value=0.0, step=1.0)
mrg = st.number_input("MRG", min_value=0.0, step=1.0)
regularity = st.number_input("Regularity", min_value=0.0, step=1.0)
top_pack = st.selectbox("Top Pack", ['NONE', 'Voice', 'Data'])  # example categories
freq_top_pack = st.number_input("Freq Top Pack", min_value=0.0, step=1.0)

if st.button("Predict"):
    # Convert categorical variables to numeric if needed (simplified example)
    region_map = {'DAKAR': 0, 'THIES': 1, 'DIOURBEL': 2, 'FATICK': 3}
    tenure_map = {'0-3 month': 0, '3-6 month': 1, '6-9 month': 2, '9-12 month': 3, '12-15 month': 4,
                  '15-18 month': 5, '18-21 month': 6, '21-24 month': 7, 'K > 24 month': 8}
    top_pack_map = {'NONE': 0, 'Voice': 1, 'Data': 2}

    input_data = np.array([
        region_map[region],
        tenure_map[tenure],
        montant,
        frequence_rech,
        revenue,
        arpu_segment,
        frequence,
        data_volume,
        on_net,
        orange,
        tigo,
        zone1,
        zone2,
        mrg,
        regularity,
        top_pack_map[top_pack],
        freq_top_pack
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("Prediction: The customer is likely to **churn**.")
    else:
        st.success("Prediction: The customer is likely to **stay**.")
