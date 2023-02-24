import streamlit as st
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder # Ordinal encoder

st.title("Play Golf???")
#streamlit run app2.py

temp_dict = {'Hot': 1, 'Cool': 0,'Mild' : 2}
outlook_dict = {'Overcast': 0, 'Rainy': 1,'Sunny' : 2}
humid_dict = {'High': 0, 'Normal': 1}
windy_dict = {'True': True, 'False': False}


outlook = st.selectbox('Outlook', outlook_dict)
temp = st.selectbox('Temperature', temp_dict)
# st.write(temp)
humidity = st.selectbox('Humidity', humid_dict)
windy = st.selectbox('Windy', windy_dict)

model =  pickle.load(open('model.sav','rb'))
x = model.predict([[outlook_dict[outlook],temp_dict[temp],humid_dict[humidity],windy_dict[windy]]])
st.write(x[0])