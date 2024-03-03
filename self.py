import streamlit as st
from PIL import Image
def self_introduction():
    st.subheader("About the Project Maker :")
    st.write("- `Name:` Soumodip Khan")
    st.write("- `Qualification:` MBA in Business Analytics & Data Science")
    self_image = Image.open("Soumodip.jpg")
    scaled_image = self_image.resize((295, 291))
    st.image(scaled_image, caption='Business Analytics & Data Science', use_column_width=True)
    st.write("""Hello! I'm Soumodip Khan, a dedicated student passionate about the world of data science and its transformative potential. With a background in Business Analytics, I thrive on exploring complex datasets and uncovering actionable insights that drive innovation and efficiency. My journey in data science has been a thrilling exploration of statistical methodologies, machine learning algorithms, and data visualization techniques. I am driven by a curiosity to understand patterns in data and solve real-world problems with analytical rigor. Constantly seeking to expand my knowledge and skill set, I am excited to contribute to the ever-evolving landscape of data-driven decision-making""")
self_introduction()