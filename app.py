import streamlit as st
import numpy as np
import pandas as pd

from streamlit_option_menu import option_menu
from prediction import prediction
from visualization import generate_charts1
from about import display_project_info
from self import self_introduction
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load the model




# Main function
st.title("Mobile Price Prediction Project")

    # Button to show project information


# Streamlit app



# Input widgets

selected = option_menu(
                menu_title=None,  # required
                options=[ 'Project info',"Visual Analysis","Predict Price","About the Project Maker"],  # required
                icons=[ "info-circle-fill","info-circle-fill","clipboard-data-fill","file-person"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
                orientation="horizontal",
            )
if selected=='Project info':
    st.subheader("Page No: 1")
    display_project_info()

if selected == "Visual Analysis":
    st.subheader("Page No: 2")
    generate_charts1()

if selected == "Predict Price":
    st.subheader("Page No: 3")
    prediction()
if selected == "About the Project Maker":
    st.subheader("Page No: 5")
    self_introduction()
