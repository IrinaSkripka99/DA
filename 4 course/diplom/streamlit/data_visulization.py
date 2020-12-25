import time

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import tree

from config import train_data, test_cols, home_image


# page functioning
def app():
    html1 = '''
            <style>
            #heading{
              color: #42617d;
              text-align:top-left;
              font-size: 45px;
            }
            </style>
            <h1 id = "heading"> Ad Data Analyses</h1>
        '''
    st.markdown(html1, unsafe_allow_html=True)

    def mem_usage(pandas_obj):
        if isinstance(pandas_obj, pd.DataFrame):
            usage_b = pandas_obj.memory_usage(deep=True).sum()
        else:  # we assume if not a df it's a series
            usage_b = pandas_obj.memory_usage(deep=True)
        usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
        return "{:03.2f} MB".format(usage_mb)

    col1, col2 = st.beta_columns(2)
    # button functionality for data points
    param1 = col1.selectbox("Select the X Parameter", test_cols)

    param2 = col2.selectbox("Select the Y Parameter", test_cols)

    df_train_first_10k = pd.read_csv(train_data, sep='|', nrows=10000)

    data = df_train_first_10k
    spec = {
        "mark": "line",
        "encoding": {
            "x": {"field": param1, "type": "quantitative"},
            "y": {"field": param2, "type": "quantitative"},
        },
    }

    st.subheader("View dependence on two variables")
    plt_2_variables = st.vega_lite_chart(spec, width=500, height=300)
    plt_2_variables.vega_lite_chart(data, spec)

    st.subheader("View the dependence of a variable on the number of values")
    param3 = st.selectbox("Select the parameter", test_cols)

    count_df = data[param3].value_counts()
    st.bar_chart(count_df,width=1000, height=500)

    st.subheader("Corr matrix")
    corr = data.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure

    def plotCorrelationMatrix(df, graphWidth):
        filename = "df.dataframeName"
        df = df.dropna('columns')
        df = df[[col for col in df if df[col].nunique() > 1]]
        if df.shape[1] < 2:
            print(
                f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
            return
        corr = df.corr()
        plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
        corrMat = plt.matshow(corr, fignum=1)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.gca().xaxis.tick_bottom()
        plt.colorbar(corrMat)
        plt.title(f'Correlation Matrix for {filename}', fontsize=15)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(plt.show())

    plotCorrelationMatrix(data, 21)
    # time.sleep(0.2)  10 Sleep a little so the add_rows gets sent separately.
    # x.add_rows(data)
    #
    # x = st.line_chart()
    # x.add_rows(data)
    #
    # x = st.area_chart()
    # x.add_rows(data)
    #
    # x = st.bar_chart()
    # x.add_rows(data)
    #
    # st.subheader("Here is 1 empty map")
    # st.deck_gl_chart()
