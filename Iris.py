# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 09:15:03 2020

@author: karthi
"""

import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write('''
         # Simple Iris Flower Prediction App
         This app Predicts Iris Flower Type!
         ''')

st.sidebar.header("User Input Data")

def User_Input_features():
    Sepal_Length=st.sidebar.slider("Sepal Length",4.3,7.9,5.4)
    Sepal_Width= st.sidebar.slider("Sepal Width",2.0,4.4,3.4)
    Petal_Length=st.sidebar.slider("Petal Length",1.0,6.9,1.4)
    Petal_Width= st.sidebar.slider("Petal Width",0.1,2.5,0.2)
    
    data={"Sepal_Length":Sepal_Length,
          "Sepal_Width":Sepal_Width,
          "Petal_Length":Petal_Length,
          "Petal_Width":Petal_Width
        }
   
    Features=pd.DataFrame(data,index=[0])
    return Features

df=User_Input_features()
st.subheader("User Input Data")
st.write(df)

iris=datasets.load_iris()
X=iris.data
Y=iris.target

Rd=RandomForestClassifier()
Rd.fit(X,Y)

Prediction=Rd.predict(df)
Prediction_prob=Rd.predict_proba(df)

st.subheader("Types of Flowers with thier corresponding  Index Values")
st.write(iris.target_names)

st.subheader("Predicted Flower")
st.write(iris.target_names[0])

st.subheader("Prediction Probability Depending On Flower Index Value")
st.write(Prediction_prob)

page_bg_img = '''
<style>
body {
background-image: url("https://i.pinimg.com/originals/18/f6/bb/18f6bb2e4767f2a09bf74b1ada2e5de3.jpg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)