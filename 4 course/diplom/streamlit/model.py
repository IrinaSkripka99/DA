import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import tree
from config import train_data, test_data


#page functioning
def app():
     html1 = '''
                 <style>
                 #heading{
                   color: #42617d;
                   text-align:top-left;
                   font-size: 45px;
                 }
                 </style>
                 <h1 id = "heading"> Sales Data Prediction</h1>
             '''
     st.markdown(html1, unsafe_allow_html = True)

     X_train = pd.read_csv(train_data, sep='|', nrows=10000)
     Y_train = pd.read_csv(test_data, sep='|', nrows=10000)

     model = tree.DecisionTreeRegressor(max_depth=8)
     # model.fit(X_train, Y_train)
     # y = model.score(X_train, Y_train)
     st.write(X_train)
