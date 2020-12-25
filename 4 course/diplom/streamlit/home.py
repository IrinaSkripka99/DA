import pandas as pd
import streamlit as st

from config import train_fields,test_fields, home_image


# page functioning
def app():
    # heading and text information
    html1 = '''
    <style>
    #heading{
      color: #42617d;
      text-align:top-left;
      font-size: 45px;
    }
    #sub_heading1{
    color: #42617d;
    text-align: right;
    font-size: 30px;
    }
    #sub_heading2{
    color: #42617d;
    text-align: left;
    font-size: 30px;
      }
    #usage_instruction{
    text-align: right;
    font-size : 15px;
    }
    #data_info{
    text-align : left;
    font-sixe : 15px;
    }
    /* Rounded border */
    hr.rounded {
        border-top: 6px solid #42617d;
        border-radius: 5px;
    }
    </style>
    <h1 id = "heading">Advertisement CTR prediction </h1>
    Advertisement CTR prediction is the key problem in the area of computing advertising. Increasing the accuracy of Advertisement CTR prediction is critical to improve the effectiveness of precision marketing. In this competition, we release big advertising datasets that are anonymized. Based on the datasets, contestants are required to build Advertisement CTR prediction models. The aim of the event is to find talented individuals to promote the development of Advertisement CTR prediction algorithms.
    <a href = "https://www.youtube.com/watch?v=9dmLxMNgCGM" target="_blank"><i><b>more</i></b></a>
    </h3>
    '''
    st.markdown(html1, unsafe_allow_html=True)
    st.image(home_image, width=700, output_format="PNG")
    html2 = '''
    <hr class="rounded">
    <h3 id = "sub_heading1">Usage Description&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</h3>
    <p id = "usage_instruction">The UI/UX for the app glides using the <b>Sidebar</b> to the left.&emsp;&emsp;<br>
    Access all the features of the app using it.The web app comes&nbsp;<br>
    with the features including -&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp; <br>
    <b>1. Sales Analyses using <i> Data Visulations</i> based on <i>Seaborn</i><br>
    <b>2. Future Sale Prediction using <i>Decision Tree Algorithm</i>&emsp;&emsp;<br>
    <h3 id ="sub_heading2">Data Overview&emsp;&emsp;&emsp;</h3>
    <p id ="data_info">The datasets contain the <i>advertising behavior</i> data collected from seven consecutive days, including a training dataset and a testing dataset. &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</p>
    '''
    st.markdown(html2, unsafe_allow_html=True)

    df_train = pd.read_csv(train_fields)
    df_test = pd.read_csv(test_fields)

    col1_train, col2_train = st.beta_columns(2)
    col1_test, col2_test = st.beta_columns(2)

    turn_on_train = col1_train.button("Data Fields Of Train Dataset")

    turn_on_test = col1_train.button("Data Fields Of Test Dataset")

    if (turn_on_train):
        st.subheader("Train dataset")
        st.table(df_train)
    elif (turn_on_test):
        st.subheader("Test dataset")
        st.table(df_test)

    turn_off_train  = col2_train.button("Close")