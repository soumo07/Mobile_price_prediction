import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load the model
loaded_model = pickle.load(open('rf_for_Mobie_Price_prediction', 'rb'))

# Load the dataset
df = pd.read_csv("mobile_prices_2023.csv")

# Data preprocessing
df["Price in INR"] = df["Price in INR"].str.replace("₹", "").str.replace(",", "").astype("int")
df["Number of Ratings"] = df["Number of Ratings"].str.replace(",", "").astype("int")
df['Price_Range'] = 0
a = 0
for i in range(len(df['Price in INR'].values)):
    if df['Price in INR'][a] <= 7000:
        df.at[a, 'Price_Range'] = 1
    elif df['Price in INR'][a] <= 14000:
        df.at[a, 'Price_Range'] = 2
    elif df['Price in INR'][a] <= 21000:
        df.at[a, 'Price_Range'] = 3
    elif df['Price in INR'][a] <= 30000:
        df.at[a, 'Price_Range'] = 4
    elif df['Price in INR'][a] <= 45000:
        df.at[a, 'Price_Range'] = 5
    elif df['Price in INR'][a] <= 60000:
        df.at[a, 'Price_Range'] = 6
    else:
        df.at[a, 'Price_Range'] = 7
    a += 1
df.rename(columns={'Rating ?/5': 'Rating 0/5'}, inplace=True)
df["Battery"] = df["Battery"].str.replace("mAh", "")
df.fillna({'Battery': 4000}, inplace=True)
replacement_dict = {'A15 ': 4000, 'A13 ': 4000, 'A14 ': 4000, 'A16 ': 4000, 'A12 ': 4000, 'A9 ': 4000,
                    'Apple ': 4000, '1 ': 4000, '0 ': 4000, 'MediaTek ': 4000, 'Brand ': 4000, 'Unisoc ': 4000,
                    '2 ': 4000}
df['Battery'] = df['Battery'].replace(replacement_dict)
df['Battery'] = pd.to_numeric(df['Battery'], errors='coerce')
df["Battery"] = df["Battery"].astype('int')
df.fillna({'Front Camera': 'No Value Available',
           'ROM/Storage': 'No Value Available',
           'Processor': 'No Value Available',
           'Back/Rare Camera': 'No Value Available'}, inplace=True)
df['BRAND'] = df['Phone Name'].str.split().str[0]
replacement_dict = {'No': '0', '0': '0', 'Expandable': '0', 'NA': '0'}
df['ROM/Storage'] = df['ROM/Storage'].replace(replacement_dict)
df['ROM/Storage'] = pd.to_numeric(df['ROM/Storage'], errors='coerce')
df['ROM/Storage'] = df['ROM/Storage'].astype(str)
df['ROM(GB)'] = df['ROM/Storage'].str.split().str[0].astype(float)
df['RAM(GB)'] = df['RAM'].str.split().str[0]
rom_list = ['No', '0', 'Expandable', 'NA']
df['ROM(GB)'] = df['ROM(GB)'].apply(lambda x: 0 if x in rom_list else x)
ram_list = ['NA', 'Expandable', '0.53', '3.81', 'cm', '0.046875', '1.5']
df['RAM(GB)'] = df['RAM(GB)'].apply(lambda x: 0 if x in ram_list else x)
df['RAM(GB)'] = df['RAM(GB)'].replace({'0.53': '1', '3.81': '4', '0.046875': '0', '1.5': '2'}).astype(int)
df['Main_Back_Camera'] = df['Back/Rare Camera'].str.split().str[0].str.replace("MP", "").str.replace("Mp", "").replace('No', 0).astype(float)
df['Main_Front_Camera'] = df['Front Camera'].str.split().str[0].str.replace("MP", "").str.replace("Mp", "").replace('No', 0).astype(float)
BackCamera_list = [camera.split("+") for camera in df['Back/Rare Camera']]
df['Back_camera_Score'] = [len(sublist) for sublist in BackCamera_list]
frontCamera_list = [camera.split("+") for camera in df['Front Camera']]
df['front_camera_Score'] = [len(sublist) for sublist in frontCamera_list]
df['BRAND'] = df['BRAND'].replace({"APPLE": "Apple", "apple": "Apple", "Nexus": "Google",
                                   "realme": "Realme", "vivo": "Vivo", "SAMSUNG": "Samsung",
                                   "10A": "Samsung", "Mi": "Xiaomi", "MOTOROLA": "Motorola","Redmi": "Xiaomi","REDMI": "Xiaomi",
                                   "Moto": "Motorola", "A10E": "Samsung", "a": "Samsung", "�9A": "Samsung", "10A": "Samsung"})
le = LabelEncoder()
df['Encoding_Brand'] = le.fit_transform(df['BRAND'])
# Prepare data for modeling
train_data, test_data = train_test_split(df, test_size=0.25, random_state=42)
train_data_model = train_data[['Encoding_Brand', 'Number of Ratings','Battery', 'Rating 0/5', 'ROM(GB)',
                               'RAM(GB)', 'Back_camera_Score', 'front_camera_Score', 'Main_Back_Camera',
                               'Main_Front_Camera', 'Price in INR']]

train_data_model_X = train_data_model.iloc[:, :-1]
train_data_model_Y = train_data_model.iloc[:, -1]
brand_map = {'Alcatel':0,'Apple':1,'Google':2,'Huawei':3,'Infinix':4,'Lenovo':5,
'Motorola':6,'Nokia':7,'Nothing':8,'OPPO':9,'OnePlus':10,'POCO':11,'Realme':12,'Samsung':13,'Vivo':14,'Xiaomi':15}
def prediction():
    st.title('Prediction')

    BRAND = (st.selectbox('***Brand***', (df['BRAND'].unique())))
    num_ratings = st.slider('***Number of Ratings(by No of person)***', 0, 1500000,1000)
    Battery = st.number_input('***Battery***', min_value=0, step=1)
    rating = st.number_input('***Rating(Range-(0-5))***', min_value=0.0, max_value=5.0, step=0.1)
    rom = st.number_input('***ROM(GB)***', min_value=0, step=1)
    ram = st.number_input('***RAM(GB)***', min_value=0, step=1)
    back_cam_score = st.number_input('***No of Back Camera***', min_value=0, step=1)
    front_cam_score = st.number_input('***No of Front Camera***', min_value=0, step=1)
    Main_Back_Camera= st.selectbox('***Main Back Camera***', np.sort(df['Main_Back_Camera'].unique()))
    Main_Front_Camera=st.selectbox('***Main Front Camera***', np.sort(df['Main_Front_Camera'].unique()))
    # Predict the price
    Brand_encoded = brand_map[BRAND]
    input_data = [[Brand_encoded,num_ratings,Battery, rating, rom, ram, back_cam_score, front_cam_score,Main_Back_Camera, Main_Front_Camera]]
    input_df = pd.DataFrame({
    'Value': [BRAND, num_ratings, Battery, rating, rom, ram, back_cam_score, front_cam_score,
              Main_Back_Camera,Main_Front_Camera]},
        index=['BRAND', 'Number of Ratings(by No of person)', 'Battery', 'Rating(Range-(0-5))',
           'ROM(GB)', 'RAM(GB)', 'No of Back Camera', 'No of Front Camera','Main Back Camera',
           'Main Front Camera']).T
    st.table(input_df)

# Display the predicted price
    if st.button("Predict Price"):
        predicted_price = loaded_model.predict(input_data)[0]
        st.write(f"Receive an estimated price for your anticipated phone: ₹{predicted_price:.2f}")

prediction()