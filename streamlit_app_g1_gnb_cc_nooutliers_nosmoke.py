# General Libraries
import pickle
import pandas as pd

# Model deployment
from flask import Flask
import streamlit as st

import numpy as np

model = pickle.load(open('Sprint_gnb_cc_no smoke_no outliers.pkl', 'rb'))    
scaler=pickle.load(open('scaler.pkl', 'rb'))  

st.title("Stroke Detection")
html_temp = """
<div style="background:#025246 ;padding:10px">
<h2 style="color:white;text-align:center;"> Stroke Detection ML App </h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)



#age-----
age=st.slider('Age', 0, 100, 35)


#bmi-----
bmi=st.slider('BMI (Body Mass Index)', 0, 50, 18)

#gender-----
gender=st.radio('Gender',('M','F'))
if gender == 'M':
    female=0
    male=1
else:
    female=1
    male=0

#glucose-----
glucose=st.slider('Average Glucose Level (mg/dL)', 50, 150, 90)

#heart disease-----
heart=st.radio('With Heart Disease?',('Yes','No'))
if heart == 'Yes':
    heart=1
else:
    heart=0
    
    
#hypertension-----
hypertension=st.radio('With Hypertension?',('Yes','No'))
if hypertension == 'Yes':
    hypertension=1
else:
    hypertension=0




def predict_if_stroke(age,hypertension,heart,glucose,bmi,female,male):       

    
    input_list=[age,hypertension,heart,glucose,bmi,female,male]
    transform_list=scaler.transform(np.array(input_list).reshape(1,-1))
    prediction_num=model.predict(transform_list)[0]
    pred_map = {1: 'Stroke', 0: 'No Stroke'}
    prediction = pred_map[prediction_num]
    
    return prediction

if st.button("Predict"):
    
    pred = predict_if_stroke(age,hypertension,heart,glucose,bmi,female,male)
    
    if pred == 'Stroke':
        st.error('This person might have STROKE')
    elif pred == 'No Stroke':
        st.success('This person is not at risk of STROKE')
        
        
        
        

