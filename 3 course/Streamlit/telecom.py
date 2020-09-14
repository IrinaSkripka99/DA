from __future__ import print_function, division, unicode_literals, absolute_import
from streamlit.compatibility import setup_2_3_shims

setup_2_3_shims(globals())

import streamlit as st
from streamlit import config
import altair as alt
import pydeck as pdk
import time
from streamlit import config
import random
from io import BytesIO
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
# %matplotlib inline
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import graphviz as graphviz
import pandas as pd
from scipy.io import wavfile
import numpy as np
import altair as alt

from PIL import Image, ImageDraw

from streamlit.widgets import Widgets

st.title("Telecom churn")

"Рынок телекоммуникаций в США насыщен, а темпы роста клиентов низкие. Поэтому они концентрируют внимание игроков на удержании и контроле оттока. Этот проект исследует набор данных оттока для определения ключевых факторов оттока и создает лучшую прогностическую модель для прогнозирования оттока. Представлена стратегия уменьшения оттока, и предложение оценивается по модели."
"Набор данных telecom_churn содержит 21 переменную и 3,333 наблюдения. Ниже перечислены переменные из этого набора."
"Содержимое набора данных:"

"-State: Код штатов США."
"-Account length: длина аккаунта."
"-Area code: Код города абонента."
"-Phone number: Номер телефона."
"-International plan: Подписка на международный план."
"-Voice mail plan: Подписка на план голосовой почты."
"-VMail Message: количество сообщений голосовой почты."
"-Total day minutes: количество минут звонков в течение дня."
"-Total day calls: количество звонков в течение дня."
"-Total day charge: плата в течение дня."
"-Eve Mins: количество минут звонков в течение вечера."
"-Eve Calls: количество звонков в течение вечера."
"-Eve Charge: Заряжается в течение вечера."
"-Total night minutes: количество минут звонков в течение ночи."
"-Total night calls: количество звонков в течение ночи."
"-Total night charge: зарядка ночью."
"-Intl Mins: количество минут международных вызовов."
"-Intl Calls: количество международных звонков."
"-Intl Charge: плата за международные звонки."
"-CustServ Calls: звонки в службу поддержки клиентов."
"-Churn : отток, да или нет"

with st.echo():
    # Text input
    test_size = st.sidebar.slider("Test size", 0.0, 1.0, 0.25)
    cv = st.sidebar.slider("cv", 0, 10, 5)
    st.subheader("Загружаем библиотеки & Данные")
    "Имеется ли у вас cvs файл?"
    yes = st.button("Yes")
    no = st.button("No")

with st.echo():
    if no:
        df = pd.read_csv("./telecom_churn.csv", low_memory=False)
    if yes:
        w10 = st.file_uploader("Upload a CSV file", type="csv")
        if w10:
            df = pd.read_csv(w10)
if yes or no:
    with st.echo():
        df.isnull().any().any()
        st.write(df)
    with st.echo():
        # Изменить тип колонки
        df["churn"] = df["churn"].astype("int64")
        # Чтобы посмотреть статистику по нечисловым признакам, нужно явно указать интересующие нас типы в параметре include
        st.write(df.describe(include=["object", "bool"]))
    with st.echo():
        # берем конкретное значение
        numerical = list(
            set(df.columns)
            - set(
                [
                    "state",
                    "international plan",
                    "voice mail plan",
                    "area code",
                    "churn",
                    "customer service calls",
                ]
            )
        )
        # Выводим корреляцию каждой ячейки
        corr_matrix = df[numerical].corr()
        st.write(corr_matrix)

    "Как видно из приведенного выше графика корреляции, мы можем заметить, что есть 4 переменные (total day charge, total eve charge, total night charge, total intl charge), которые напрямую зависят от (total day call, total eve calls, total night calls, total intl calls). Они называются зависимыми переменными и могут быть опущены, поскольку они не вносят никакой дополнительной информации. так что давайте отбросим их:"
    with st.echo():
        # Убираем незначимые признаки
        numerical = list(
            set(numerical)
            - set(
                [
                    "total day charge",
                    "total eve charge",
                    "total night charge",
                    "total intl charge",
                ]
            )
        )
        corr_matrix = df[numerical].corr()
        # Надо найти тепловую карту для стремлита  я хз
        st.write(corr_matrix)
    with st.echo():
        # Удаляем из наших данных
        df.drop(
            [
                "total day charge",
                "total eve charge",
                "total night charge",
                "total intl charge",
            ],
            axis=1,
        )

    "Информация об оттоке"
    with st.echo():
        st.write(df.groupby("churn")["phone number"].count())

    "Индексирование и извлечение данных оттока пользователей в наших данных"
    with st.echo():
        # Выводим среднее значение оттока
        st.write(df["churn"].mean())
    with st.echo():
        # Выводим среднее значение каждого признака с положительным оттоком
        st.write(df[df["churn"] == 1].mean())

    "Посмотрим на распределение пользователей по переменной churn. Укажем значение параметра normalize=True, чтобы посмотреть не абсолютные частоты, а относительные"
    with st.echo():
        st.write(df["churn"].value_counts(normalize="True"))

    with st.echo():
        # Сколько времени (в среднем) ушедшие пользователи проводят в телефоне в дневное время
        st.write(df[df["churn"] == 1]["total day minutes"].mean())
    with st.echo():
        # Среднее значение признаков,которые не подключены к международному тарифу и с отрицательным оттоком
        st.write(
            df[(df["churn"] == 0) & (df["international plan"] == "no")][
                "total day minutes"
            ].mean()
        )

    "Спрогнозируем отток"
    "Какова связь между оттоком и международным планом?"
    with st.echo():
        pd.crosstab(df["churn"], df["international plan"], margins=True, normalize=True)

        # f = plt.figure()
        # x = df['international plan']
        # y=df['churn']
        # plt.hist(x,y)

        # st.plotly_chart(f)

        # pd.crosstab(df['churn'], df['state'], margins = True)
        # sns.countplot(x = df['state'], hue = 'churn', data = df);

        # pd.crosstab(df['churn'], df['number vmail messages'], margins = True)
        # sns.countplot(x = df['number vmail messages'], hue = 'churn', data = df);

        # Тут еще код нада старый

    "Разделение набора данных на обучающие и тестовые данные в соответствии с необходимыми измерениями"

    with st.echo():
        drp = df[
            [
                "state",
                "area code",
                "phone number",
                "international plan",
                "voice mail plan",
                "churn",
            ]
        ]
        X = df.drop(drp, 1)
        y = df.churn
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size
        )

    "Примененим логистическую регрессию для прогнозирования переменной оттока."
    "Также измеряется оценка точности алгоритма."
    with st.echo():
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        st.write(
            "Logistic regression score =",
            round(metrics.accuracy_score(y_test, y_pred), 2),
        )

    "Используем кросс-валидацию ,по умолчанию с 5 разделами данных."
    "Выводим результаты каждого раздела данных."
    "Выводим средний балл всех разделов."
    with st.echo():
        scores = cross_val_score(logreg, X, y, cv=cv, scoring="accuracy")
        st.write("Логистическая регрессия каждой части\n", scores)
        st.write(
            "Средний балл всех баллов после кросс валидации =", round(scores.mean(), 2)
        )

    # Должна быть матрица ошибок

    "Вычисление ставок по confusion матрице"
    # FP = conf[1][0]
    # FN = conf[0][1]
    # TP = conf[0][0]
    # TN = conf[1][1]
    # st.write('False Positive ',FP)
    # st.write('False Negative ',FN)
    # st.write('True Positive ',TP)
    # st.write('True Negative ',TN)
    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP/(TP+FN)
    # st.write('\nTrue Positive Rate :',round(TPR,2))
    # # Specificity or true negative rate
    # TNR = TN/(TN+FP)
    # st.write('\nTrue Negative Rate :',round(TNR,2))
    # # Precision or positive predictive value
    # PPV = TP/(TP+FP)
    # st.write('\nPositive Predictive Value :',round(PPV,2))
    # # Negative predictive value
    # NPV = TN/(TN+FN)
    # st.write('\nNegative Predictive Value :',round(NPV,2))
    # # Fall out or false positive rate
    # FPR = FP/(FP+TN)
    # st.write('\nFalse Positive Rate :',round(FPR,2))
    # # False negative rate
    # FNR = FN/(TP+FN)
    # st.write('\nFalse Negative Rate :',round(FNR,2))
    # # False discovery rate
    # FDR = FP/(TP+FP)
    # st.write('\nFalse Discovery Rate :',round(FDR,2))

    # # Overall accuracy
    # ACC = (TP+TN)/(TP+FP+FN+TN)
    # st.write('\nOverall accuracy :',round(ACC,2))
