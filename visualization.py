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



def generate_charts1():
        st.title(':bar_chart: Exploratory Data Analysis')
        st.markdown('<style>div.block-container{padding_top:1rem;}</style>', unsafe_allow_html=True)
        st.header('Chart-1')
        fig, ax = plt.subplots(figsize=(20, 15))
        sns.heatmap(train_data_model.corr(), cmap=sns.cubehelix_palette(as_cmap=True), annot=True, linewidth=.5)
        plt.title('Correlation Heatmap')
        plt.xlabel('Features')
        plt.ylabel('Features')
        st.pyplot(fig)

        st.header('Chart-2')
        fig, ax = plt.subplots(figsize=(20, 15))
        palette = sns.color_palette("bright", len(df['Rating 0/5'].unique()))
        sns.barplot(x=df['Rating 0/5'], y=df['Battery'], palette=palette)
        plt.title('Rating vs Battery')
        plt.xlabel('Rating')
        plt.ylabel('Battery')
        fig.suptitle('Rating 0/5 vs Battery', fontsize=20)
        st.pyplot(fig)

        st.header('Chart-3')
        fig, ax = plt.subplots(1, 2, figsize=(18, 7))
        palette = sns.color_palette("bright", len(df['Price_Range'].unique()))
        sns.barplot(x=df['Price_Range'], y=df['Battery'], ax=ax[0], palette=palette)
        ax[0].bar_label(ax[0].containers[0])
        sns.lineplot(x=df['Price_Range'], y=df['Battery'], ax=ax[1], palette=palette)
        ax[1].set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
        fig.suptitle('Price_Range vs Battery', fontsize=20)
        st.pyplot(fig, use_container_width=True)

        st.header('Chart-4')
        fig, ax = plt.subplots(1, 2, figsize=(18, 7))
        palette = sns.color_palette("bright", len(df['Price_Range'].unique()))
        sns.barplot(x=df['Price_Range'], y=df['Back_camera_Score'], ax=ax[0], palette=palette)
        ax[0].bar_label(ax[0].containers[0])
        sns.lineplot(x=df['Price_Range'], y=df['Back_camera_Score'], ax=ax[1], palette=palette)
        ax[1].set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
        fig.suptitle('Price_Range vs Back_camera_Score', fontsize=20)
        st.pyplot(fig)

        st.header('Chart-5')
        fig, ax = plt.subplots(1, 2, figsize=(18, 7))
        palette = sns.color_palette("bright", len(df['Price_Range'].unique()))
        sns.barplot(x=df['Price_Range'], y=df['Main_Front_Camera'], ax=ax[0], palette=palette)
        ax[0].bar_label(ax[0].containers[0])
        sns.lineplot(x=df['Price_Range'], y=df['Main_Front_Camera'], ax=ax[1], palette=palette)
        ax[1].set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
        fig.suptitle('Price_Range vs Main_Front_Camera', fontsize=20)
        st.pyplot(fig, use_container_width=True)
        st.header('Chart-6')
        fig, ax = plt.subplots(1, 1, figsize=(18, 7))
        sns.barplot(y=df['Number of Ratings'], x=df['Price_Range'], hue=df['Price_Range'])
        fig.suptitle('Price_Range vs Number of Ratings', fontsize=20)
        st.pyplot(fig)
        st.header('Chart-7')
        fig, ax = plt.subplots(1, 2, figsize=(18, 7))
        palette = sns.color_palette("bright", len(df['Price_Range'].unique()))
        sns.barplot(x=df['Price_Range'], y=df['RAM(GB)'], ax=ax[0], palette=palette)
        ax[0].bar_label(ax[0].containers[0])
        sns.lineplot(x=df['Price_Range'], y=df['RAM(GB)'], ax=ax[1], palette=palette)
        ax[1].set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
        fig.suptitle('Price_Range vs RAM(GB)', fontsize=20)
        st.pyplot(fig)
        st.header('Chart-8')
        fig, ax = plt.subplots(1, 2, figsize=(25, 7))
        palette = sns.color_palette("bright", len(df['Encoding_Brand'].unique()))
        sns.barplot(x=df['Encoding_Brand'], y=df['Number of Ratings'], ax=ax[0], palette=palette)

        sns.lineplot(x=df['Encoding_Brand'], y=df['Number of Ratings'], ax=ax[1], palette=palette)
        ax[0].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        fig.suptitle('Encoding_Brand vs Number of Ratings', fontsize=20)
        st.pyplot(fig)
        st.subheader("Insights")
        st.write(
            "***Correlation Analysis (Chart-1)***: The correlation analysis revealed that features like ROM(GB), RAM(GB), Battery, Brand, and Camera are significant factors influencing the price of a mobile phone.")

        st.write(
            "***Rating vs. Battery Power (Chart-2)***: Phones with higher ratings tend to have higher battery power, with a peak observed at a rating of 4.8.")

        st.write(
            "***Battery Power across Price Ranges (Chart-3)***: Mobile phones in price range 2 (7000 to 14000) exhibit higher battery power compared to other price ranges.")

        st.write(
            "***Back Camera Score vs. Price Range (Chart-4)***: As the price range increases, the back camera score also tends to increase.")

        st.write(
            "***Main Front Camera Score vs. Price Range (Chart-5)***: The main front camera score peaks at price range 2, indicating that phones within this range offer the best front camera quality.")

        st.write(
            "***Rating Distribution across Price Ranges (Chart-6)***: The majority of ratings are given to phones in the price range of 7000 to 14000.")

        st.write(
            "***RAM Distribution across Price Ranges (Chart-7)***: Price range 7 has the highest RAM capacity, followed by price range 6 and then price range 5.")

        st.write(
            "***Brand Ratings (Chart-8)***: Xiaomi receives the highest ratings compared to other brands, indicating its popularity among consumers.")

        st.write(
            "By reframing the text, the insights from each chart are presented clearly and concisely, making it easier for the reader to understand the findings")

generate_charts1()