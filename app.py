import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import pandas as pd
import joblib

# Load the model
rf_model = joblib.load('rf_US_Admission_model.joblib')

# with open('rf_US_Admission_model.pkl', 'rb') as model_file:
#     rf_model = pickle.load(model_file)

feature_names = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research']

st.title("US Masters Admission Prediction")
st.write("This model will help you predict the likelyhood of getting admission based on supporting factors.")

GRE = st.number_input("GRE Score",0,340)
TOEFL = st.number_input("TOEFL Score",0,120)
University = st.number_input("University Rating",0,5)
SOP = st.number_input("Statement of Purpose",0.1,5.0)
LOR = st.number_input("Letter of Recommendation Strength",0.1,5.0)
GPA = st.number_input("Undergraduate GPA",0.1,10.0)
Research = st.number_input("Research Score",0,1)


input_data = pd.DataFrame([[GRE,TOEFL,University,SOP, LOR,GPA,Research]], columns=feature_names)

if st.button("Predict"):
    prediction = rf_model.predict(input_data)

    if prediction[0] == 1:
        st.success("The model forecasts a positive outcome: the student is likely to be admitted.", icon="✅")
    else:
        st.warning("The model forecasts a positive outcome: the student is unlikely to be admitted.",icon="⚠️")    