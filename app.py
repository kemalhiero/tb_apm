import streamlit as st
import numpy as np
import pandas as pd
from preprocess import preprocess

# st.title('Klasifikasi film berdasarkan genre')

# st.write("""
#     ### Ketikkan ulasan film yang anda sukai
# """)

data = pd.read_csv('data/imdb_top_1000.csv')
# st.dataframe(data)

# form = st.form(key='my-form')
# ulasan = form.text_area('Ulasan film', '')
# submit = form.form_submit_button('Submit')