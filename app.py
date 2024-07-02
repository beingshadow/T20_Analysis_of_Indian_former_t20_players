import streamlit as st
from virat1 import virat_title,virat_kohli_performance
from dhoni import dhoni_performance,dhoni_title
from Yuvi import yuvi_title,yuvi_performance
from rohit import rohit_performance,rohit_title

player_name = st.sidebar.selectbox(
    'Select Player Name',
    ('Virat Kohli', 'Rohit Sharma', 'MS Dhoni','Yuvraj Singh')
)

if player_name=='Virat Kohli':
    virat_title()
    virat_kohli_performance()

elif player_name=="Rohit Sharma":
    rohit_title()
    rohit_performance()

elif player_name=="MS Dhoni":
    dhoni_title()
    dhoni_performance()

elif player_name=="Yuvraj Singh":
    yuvi_title()
    yuvi_performance()

