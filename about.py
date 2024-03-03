import streamlit as st
def display_project_info():
    st.header("Project Information")

    # Project Description
    st.subheader("Project Objective:")
    st.write("In the realm of purchasing mobile phones, determining the precise price can often be challenging. This application aims to alleviate this issue by providing users with an estimated price for their ideal phone, based on various specifications such as back camera quality, front camera resolution, RAM, ROM, user ratings, and battery capacity")

    # Objectives
    st.subheader("Functionality:")
    st.write("The application operates by leveraging data from 2023 on mobile phone prices. Initially, the data undergoes thorough preprocessing, exploratory data analysis (EDA), and cleaning processes. Subsequently, predictive models are developed, employing both linear regression and random forest regression techniques to address the regression problem effectively.")

    # Methods
    st.subheader("Prediction Accuracy")
    st.write("The models exhibit commendable performance, with a training accuracy of 93% and a testing accuracy of 89%. Notably, the random forest regressor model yields the highest accuracy, ensuring reliable price estimations")

    st.subheader("Application Overview")
    st.write("The application comprises three main sections. Firstly, there's an introduction providing insights into the project's goals, objectives, and the problem statement it addresses. Secondly, users can delve into the exploratory data analysis, gaining valuable insights from various charts and visualizations. Finally, the core functionality of the app lies in the prediction section, where users input their desired phone specifications to obtain an approximate price prediction.")
    # Custom CSS for multicolor background
    st.subheader("Experience the Application:")
    st.write("Discover the inner workings of this project and utilize it to forecast the price of your dream mobile phone")
display_project_info()