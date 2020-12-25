import streamlit as st

import home
import data_visulization
import model
from config import nav_image

#Pages in the app
PAGES = {
    "Home": home,
    "Data Visualization": data_visulization,
    "Data Prediction": model
}

#background styling
page_bg = '''
<style>
body {
background-color : #f4f4f4;
}
</style>
'''
st.markdown(page_bg, unsafe_allow_html=True)

#navbaar styling
st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#292929,#42617d;9);
    color: black;
    align-text: center;
}
hr.rounded {
        border-top: 6px solid #42617d;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True,
)

#inseting image in the sidebar
st.sidebar.image(nav_image, use_column_width=True)

#navbar content-1
html3 = '''
<h2 style="text-align: center;">HUAWEI DIGIX Global Challenge Competiton A</h2>
<p style="text-align: center; font-size: 15px"><i>2020 DIGIX Advertisement CTR Prediction</i></p>
<hr class="rounded">
'''
st.sidebar.markdown(html3, unsafe_allow_html=True)

st.sidebar.title("Explore")

#radio selection for the pages
selection = st.sidebar.radio("", list(PAGES.keys()))
page = PAGES[selection]
page.app()
if(page == model):
    hour = st.sidebar.number_input("hour", 0, 23, 10)